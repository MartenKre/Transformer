import cv2
import os
import numpy as np
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import threading
import time
from collections import defaultdict
from scipy.optimize import linear_sum_assignment, minimize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utility.Transformations import ECEF2LatLng, T_ECEF_Ship, LatLng2ECEF, haversineDist
import utility.Transformations as T
from util.association_utility import filterBuoys, createQueryData
from utility.GeoData import GetGeoData
from utility.Rendering import RenderAssociations
from util.init_model import init_V2W_Transformer
from util.plot_utils import plot_one_box
from models.detr import PostProcess
from boxmot import ByteTrack
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class BuoyAssociation():    
    def __init__(self, device_model="cuda", img_sz=[1920, 1080], path_to_weights="training_results/run_MLP10_newMet/best.pth"):
        self.image_size = img_sz
        self.resize_coeffs = [0.5, 0.5] # applied to video frame prior to inference, 0 -> width, 1 -> height 
        self.conf_thresh = 0.9          # BBs below this threshhold won't be plotted

        self.model = init_V2W_Transformer(lr_backbone=1e-5, hidden_dim=256, dropout=.1, nheads=8, dim_feedforward=2048,
                                          enc_layers=6, dec_layers=6, pre_norm=True, aux_loss=False, use_embeddings=False)
        self.device = torch.device(device_model) if torch.cuda.is_available() else torch.device('cpu')
        self.loadWeights(path_to_weights)
        self.model.eval()

        self.postprocess = PostProcess()
        self.BuoyCoordinates = GetGeoData(tile_size=0.02) # load BuoyData from GeoJson

        self.imu_data = None
        self.RenderObj = None   # render Instance
        self.track_buffer = 60   # after exceeding thresh (frames count) a lost BB will be reassigned new ID 
        self.MOT = self.initBoxMOT()        # Multi Object Tracker Instance
        self.color_dict = {}


    def loadWeights(self, path_to_weights):
        if not os.path.isfile(path_to_weights):
            raise ValueError("Invalid Path to Model weights! Check path_to_weights")
        print("Loading Model Weights...")
        checkpoint = torch.load(path_to_weights, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.to(self.device)

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Number of params:', n_parameters)


    def getPredictions(self, img, queries, queries_mask, frame_id):
        """Function runs inferense, returns Yolov7 preds concatenated with bearing to Objects & ID of BoxMOT
        Prediction dict contains ship pose and predicted buoy positions (in lat,lng)
        Args:
            Img:          pixel array (already resized & normalized)
            Queries:      Queries Tensor for nearby buoys containing dist and angle
            Queries_mask: Mask for relevant queries (torch.ones tensor in this case)
        Returns:
            outputs: dict containing preds (objectness and bounding boxes)
        """

        with torch.no_grad():
            outputs = self.model(img, queries, queries_mask)
        outputs = self.postprocess(outputs, target_size=[self.image_size[1], self.image_size[0]])  # convert boxes from cxcyhw -> xyxy w.r.t. original image size
        
        # BoxMOT tracking on predictions
        # preds_boxmot_format = pred[:,0:6]    # preds need to be in format [xyxy, conf, classID]
        # preds_boxmot_format = np.array(preds_boxmot_format)
        # res = self.MOT.update(preds_boxmot_format, img)   # res --> M X (x, y, x, y, id, conf, cls, ind)
        # res = torch.from_numpy(res)
        # ids = -1*torch.ones(size=(pred.size()[0], 1))   # default case -1
        # if len(res) > 0:
        #     ids[res[:,-1].to(torch.int32), 0] = res[:, 4].to(torch.float)
        # pred = torch.cat((pred, ids), dim=-1)   # pred = [N x [xyxy,conf,classID,dist,angle,ID]]

        return outputs


    def BuoyLocationPred(self, frame_id, preds):
        """for each BB prediction function computes the Buoy Location based on Dist & Angle of the tensor
        Args:
            frame_id: ID of current frame
            preds: prediction tensor of yolov7 (N,8) -> [Nx[xyxy, conf, cls, dist, angle]]
        Returns:
            Dict{"ship:"[lat,lng,heading], "buoy_prediction":[[lat1,lng1],[lat2,lng2]]}
        """

        latCam = self.imu_data[frame_id][3]
        lngCam = self.imu_data[frame_id][4]
        if self.use_biases:
            heading_bias = self.computeEma(self.heading_bias)
            heading = self.imu_data[frame_id][2] - np.rad2deg(heading_bias)
        else:
            heading = self.imu_data[frame_id][2]

        # trasformation:    latlng to ecef, ecef to enu, enu to ship
        x, y, z = LatLng2ECEF(latCam, lngCam)  # ship coords in ECEF
        ECEF_T_Ship = T_ECEF_Ship(x,y,z,heading)   # transformation matrix between ship and ecef

        # compute 2d points (x,y) in ship cs, (z=0, since all objects are on water surface)
        buoysX = (torch.cos(preds[:,7]) * preds[:,6]).tolist()
        buoysY = (torch.sin(preds[:,7]) * preds[:,6]).tolist()
        buoy_preds = list(zip(buoysX, buoysY))

        # transform buoyCoords to lat lng
        buoysLatLng = []
        for buoy in buoy_preds:
            p = ECEF_T_Ship @ np.array([buoy[0], buoy[1], 0, 1])    # buoy coords in ecef
            lat, lng, alt = ECEF2LatLng(p[0],p[1],p[2])
            buoysLatLng.append((lat, lng))

        return {"buoy_predictions": buoysLatLng, "ship": [latCam, lngCam, heading]}

    def getLabelsData(self, labels_dir, image_path):
        labelspath = os.path.join(labels_dir, os.path.basename(image_path) + ".json")
        if os.path.exists(labelspath):
            return self.distanceEstimator.LabelsJSONFormat(labelspath)
        else:
            print(f"LablesFile not found: {labelspath}")
            return []
        
    def getIMUData(self, path):
        # function returns IMU data as list

        if os.path.isfile(path):
            result = []
            with open(path, 'r') as f:
                data = f.readlines()
                for line in data:
                    content = line.split(",")
                    line = [float(x) for x in content]
                    result.append(line)
        else:
            files = os.listdir(path)
            filename = [f for f in files if f.endswith('.txt')][0]
            path = os.path.join(path, filename)
            result = []
            with open(path, 'r') as f:
                data = f.readlines()
                for line in data:
                    content = line.split(",")
                    line = [float(x) for x in content]
                    result.append(line)
            if len(result) == 0:
                print("No IMU data found, check path: {path}")
        return result
    
    def initBoxMOT(self):
        return ByteTrack(
            track_thresh=2 * self.conf_thresh,      # threshold for detection confidence -> seperates BBs into high and low confidence
            match_thresh=0.99,                  # matching thresh -> controls max dist allowed between tracklets & detections for a match
            track_buffer=self.track_buffer      # number of frames to keep a track alive after it was last detected
        )


    def create_run_directory(self, base_name="run", path=""):
        i = 0
        while True:
            folder_name = f"{base_name}{i if i > 0 else ''}"
            if not os.path.exists(os.path.join(path, folder_name)):
                path_to_folder = os.path.join(path, folder_name)
                os.makedirs(path_to_folder)
                print(f"Created directory: {path_to_folder} to store plots")
                return path_to_folder
            i += 1


    def displayFPS(self, frame, prev_frame_time):
        # function displays FPS on frame

        font = cv2.FONT_HERSHEY_DUPLEX
        new_frame_time = time.time() 
        fps= 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = int(fps) 
        fps = str(fps) 
        cv2.putText(frame, fps, (10, 15), font, 0.5, (50, 50, 50)) 
        return new_frame_time


    def draw_boxes(self, frame, boxes, confs, colors):
        # draws BBs that are above self.conf_thresh
        for bb,conf,clr in zip(boxes,confs,colors):
            txt = str(round(conf.item(), 3))
            if conf > self.conf_thresh:
                clr = [x*255 for x in clr]
                clr = [clr[2], clr[1], clr[0], clr[3]]
                plot_one_box(bb, frame, color=clr, label=txt)

    
    def Coords2Hash(self, coords):
        # takes a tuple of lat, lng coordinates and returns a unique hash key as string
        return str(coords[0])+str(coords[1])


    def get_colors(self, filtered_buoys):
        color_table = [(255/255, 255/255, 0, 1),
                    (102/255, 0, 102/255, 1),
                    (0, 255/255, 255/255, 1),
                    (255/255, 153/255, 255/255, 1),
                    (153/255, 102/255, 51/255, 1),
                    (255/255,153/155, 0, 1),
                    (224/255, 224/255, 224/255, 1),
                    (128/255, 128/255, 0, 1)]

        color_arr = []
        for i, x in enumerate(filtered_buoys):
            x = self.Coords2Hash(x)
            if x in self.color_dict:
                color_arr.append(self.color_dict[x])
            else:
                clr = color_table[i%len(color_table)]
                self.color_dict[x] = clr
                color_arr.append(clr)
        return color_arr

    
    def video(self, video_path, imu_path, rendering=False):
        # run buoy association on video

        if not rendering:
           self.processVideo(video_path, imu_path, rendering) 
        else:
            # initialize Rendering Framework with data
            lock = threading.Lock()
            self.RenderObj = RenderAssociations(lock, parent=self)
            self.imu_data = self.getIMUData(imu_path)
            lat_start = self.imu_data[0][3]
            lng_start = self.imu_data[0][4]
            heading_start = self.imu_data[0][2]
            self.RenderObj.initTransformations(lat_start, lng_start, heading_start) # initialize Transformation Matrices with pos & heading of first frame
            # start thread to run video processing 
            processing_thread = threading.Thread(target=self.processVideo, args=(video_path, imu_path, rendering, lock), daemon=True)
            processing_thread.start()
            # start rendering
            self.RenderObj.run()

    def processVideo(self, video_path, imu_path, rendering, lock=None):     
        # function computes predictions, and performs matching for each frame of video

        # load IMU data
        self.imu_data = self.getIMUData(imu_path)

        # load geodata
        ship_pose = [self.imu_data[0][3],self.imu_data[0][4],self.imu_data[0][2]]
        buoys_on_tile = self.BuoyCoordinates.getBuoyLocations(*ship_pose[0:2])
        newBuoyCoords = threading.Event()   # event that new data has arrived from thread
        results_list = []
        #self.BuoyCoordinates.plotBuoyLocations(buoyCoords)

        cap = cv2.VideoCapture(video_path)
        current_time = time.time()
        color_assignment = {}

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()

        frame_id = 0
        paused = False
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                color_dict_gt = {}
                # preprocess data (create image & query tensor)
                imu_curr = self.imu_data[frame_id]
                ship_pose = [imu_curr[3],imu_curr[4],imu_curr[2]]
                buoys_filtered = filterBuoys(ship_pose, buoys_on_tile)
                color_arr = self.get_colors(buoys_filtered)
                color_dict_gt = {i:color for i, color in enumerate(color_arr)}
                queries = torch.tensor(createQueryData(ship_pose, buoys_filtered), dtype=torch.float32)[...,0:2]

                if queries.numel() > 0: # if no queries could be generated -> skip inference
                    if queries.ndim == 1:
                        queries = queries.unsqueeze(0)
                    # normalize query inputs (dist and angle)
                    queries[..., 0] = queries[..., 0] / 1000
                    queries[..., 1] = queries[..., 1] / 180

                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (0,0), fx=self.resize_coeffs[0], fy=self.resize_coeffs[1])
                    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255    # standardizes img data & converts to 3xHxW

                    # add batch dims
                    queries = queries.unsqueeze(0).to(self.device)
                    queries_mask = torch.full((1,queries.size(dim=1)), fill_value=True).to(self.device)
                    img = img.unsqueeze(0).to(self.device)

                    # get predictions for frame
                    outputs = self.getPredictions(img, queries, queries_mask, frame_id)

                    # draw bounding boxes
                    pred_obj = outputs['objectness'].cpu().detach()
                    pred_boxes = outputs['boxes'].cpu().detach()    # filter boxes based on objectness thresh
                    self.draw_boxes(frame, pred_boxes, pred_obj, color_arr)     # draw bbs with conf as text
                else:
                    self.color_dict = {}    # reset global color dict (to reduce mem)


                # check if new buoy coords have been set by thread
                if newBuoyCoords.is_set():
                    newBuoyCoords.clear()   # clear event flag
                    buoys_on_tile = results_list   # copy results list
                    results_list = []   # clear results list
                
                # check if buoydata needs to be reloaded
                refresh = self.BuoyCoordinates.checkForRefresh(*ship_pose[0:2])
                if refresh:
                    # load new buoycoords in seperate thread 
                    print("refreshing buoy coords")
                    t = threading.Thread(target=self.BuoyCoordinates.getBuoyLocationsThreading, 
                                                args=(ship_pose[0], ship_pose[1],results_list, newBuoyCoords), daemon=True)
                    t.start()
                

                if rendering:
                    with lock:  # send data to rendering obj
                        self.RenderObj.setShipData(*ship_pose)
                        self.RenderObj.setBuoyGT(buoys_filtered, color_dict_gt)

                # display FPS
                current_time = self.displayFPS(frame, current_time)

                # Display the frame (optional for real-time applications)
                cv2.imshow("Buoy Association", frame)
                frame_id += 1

            key = cv2.waitKey(1)
            # Press 'q' to exit the loop
            if key == ord('q'):
                break

            if key == 32:
                cv2.waitKey(-1)

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

ba = BuoyAssociation()

# test_folder = "/home/marten/Uni/Semester_4/src/Trainingdata/labeled/Testdata/954_2_Pete2"
# images_dir = os.path.join(test_folder, 'images') 
# imu_dir = os.path.join(test_folder, 'imu') 
# ba.test(images_dir, imu_dir)

ba.video(video_path="/home/marten/Uni/Semester_4/src/TestData/955_2.avi", imu_path="/home/marten/Uni/Semester_4/src/TestData/furuno_955.txt", rendering=True)
# ba.video(video_path="/home/marten/Uni/Semester_4/src/TestData/videos_from_training/19_2.avi", imu_path="/home/marten/Uni/Semester_4/src/TestData/videos_from_training/furuno_19.txt", rendering=True)
# ba.video(video_path="/home/marten/Uni/Semester_4/src/TestData/22_2.avi", imu_path="/home/marten/Uni/Semester_4/src/TestData/furuno_22.txt", rendering=True)
