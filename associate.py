"""
Script to perform buoy association on a video 
"""
import torch
import os
import cv2
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from torch._prims_common import dtype_to_type
from torchvision import transforms

from models.detr import DETR, PostProcess
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine 
from util.association_utility import getIMUData, filterBuoys, createQueryData, GetGeoData, LatLng2ECEF, T_ECEF_Ship
from util.plot_utils import plot_one_box


def init_position_encoding(hidden_dim):
    N_steps = hidden_dim // 2
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    return position_embedding


def init_backbone(lr_backbone, hidden_dim, backbone='resnet50', dilation=False):
    # masks are only used for image segmentation

    position_embedding = init_position_encoding(hidden_dim)
    train_backbone = lr_backbone > 0
    return_interm_layers = False
    backbone = Backbone(backbone, train_backbone, return_interm_layers, dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def init_transformer(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )


def draw_boxes(frame, boxes, confs, colors):
    i = 0
    for bb,conf in zip(boxes,confs):
        txt = str(round(conf.item(), 3))
        clr = colors[i][1]
        clr = [x*255 for x in clr]
        clr = [clr[2], clr[1], clr[0], clr[3]]
        plot_one_box(bb, frame, color=clr, label=txt)
        i += 1
    return colors


def init_plot(two_dim = False):
    plt.ion()
    fig = plt.figure()
    if not two_dim:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
    return plt, ax


def live_plotting(plt, ax, queries, colors):
    # transfoms buoy data to ship cs and plots them

    buoys = []
    queries = queries.cpu().detach().squeeze(0)
    for dist, angle in queries:
        # renormalize dist & angle
        dist = dist * 1000
        angle = np.deg2rad(angle * 180)
        p_x = dist * np.cos(angle)
        p_y = dist * np.sin(angle)
        buoys.append([p_x, p_y])

    ax.cla()

    # Define ship shape (arrow) in the XY plane
    scaling = 30
    coords = np.asarray([[1, 0, 0], [-2, 1, 0], [-1, 0, 0], [-2, -1, 0], [1, 0, 0]])
    coords *= scaling
    poly = Poly3DCollection([coords], color=[(0,0,0.9)], edgecolor='k')
    ax.add_collection3d(poly)

    # Customize the view
    ax.set_xlim(-100, 700)
    ax.set_ylim(-400, 400)
    ax.set_zlim(0, 1)
    #ax.set_zlim(-1, 1)  # Keep it flat in the Z-axis for an XY view

    if len(buoys) > 0:
        buoys = np.asarray(buoys)
        ax.plot3D(buoys[:,0], buoys[:,1], np.zeros(len(buoys)), 'o', color = 'green')
        for i, pred in enumerate(buoys):
            if i in [e[0] for e in colors]:
                clr = [x[1] for x in colors if x[0] == i][0]
                ax.plot3D(pred[0], pred[1], np.zeros(len(buoys)), 'o', color=clr)
            ax.plot([0, pred[0]], [0,pred[1]],zs=[0,0], color='grey', linestyle='dashed')
    # Optional: Labels for clarity
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=90, azim=-180)  # Set view to make it look like a 2D XY plane

    plt.draw()


def live_plotting_2d(plt, ax, queries, colors):
    # Transform buoy data to ship coordinate system and plot them in 2D
    buoys = []
    queries = queries.cpu().detach().squeeze(0)
    
    for dist, angle in queries:
        # Renormalize dist & angle
        dist = dist * 1000
        angle = np.deg2rad(angle * 180)
        p_x = dist * np.cos(angle)
        p_y = dist * np.sin(angle)
        buoys.append([p_x, p_y])

    ax.cla()  # Clear the previous plot
    ax.grid(color="lightgray", linestyle="--")

    # Define ship shape (arrow) in the XY plane
    scaling = 20
    coords = np.array([[1, 0], [-2, 1], [-1, 0], [-2, -1], [1, 0]]) * scaling + np.array([1, 0]) * scaling
    # ax.fill(coords[:, 0], coords[:, 1], color='blue', edgecolor='k')
    ax.fill(coords[:, 1], coords[:, 0], color='blue', edgecolor='k', zorder=5)

    # Set axis limits
    # ax.set_ylim(-200, 1200)
    # ax.set_xlim(-1000, 1000)

    # Plot buoys
    if len(buoys) > 0:
        buoys = np.array(buoys)
        # ax.plot(buoys[:, 0], buoys[:, 1], 'o', color='green')  # Default green dots
        ax.plot(buoys[:, 1], buoys[:, 0], 'o', color='green')  # Default green dots

        for i, pred in enumerate(buoys):
            if i in [e[0] for e in colors]:  # If this buoy has a color override
                clr = [x[1] for x in colors if x[0] == i][0]
                # ax.plot(pred[0], pred[1], 'o', color=clr)
                ax.plot(pred[1], pred[0], 'o', markerfacecolor=(0, 0, 0, 0), markeredgecolor=clr, markersize=10, markeredgewidth=2)

                # # Draw dashed lines from origin to each buoy
                # ax.plot([0, pred[0]], [0, pred[1]], color='grey', linestyle='dashed')
                ax.plot([0, pred[1]], [0, pred[0]], color=clr, linestyle='dashed')


    # plot_fov
    fov_with_padding=110
    dist_thresh=1000
    nearby_thresh=50
    for s in [-1, 1]:
        y1 = np.sin(s * np.deg2rad(fov_with_padding//2)) * nearby_thresh
        x1 = np.cos(s * np.deg2rad(fov_with_padding//2)) * nearby_thresh
        y2 = np.sin(s * np.deg2rad(fov_with_padding//2)) * dist_thresh 
        x2 = np.cos(s * np.deg2rad(fov_with_padding//2)) * dist_thresh 
        # ax.plot([x1, x2], [y1, y2], linestyle="solid", color="grey")
        ax.plot([y1, y2], [x1, x2], linestyle="solid", color="grey")

    step = 5
    for fov in range(-1*fov_with_padding//2, fov_with_padding//2, step):
        y1 = np.sin(np.deg2rad(fov)) * dist_thresh
        x1 = np.cos(np.deg2rad(fov)) * dist_thresh
        y2 = np.sin(np.deg2rad(fov+step)) * dist_thresh 
        x2 = np.cos(np.deg2rad(fov+step)) * dist_thresh 
        ax.plot([y1, y2], [x1, x2], linestyle="solid", color="grey")

    step = 5
    for fov in range(fov_with_padding//2, 360-fov_with_padding//2, step):
        y1 = np.sin(np.deg2rad(fov)) * nearby_thresh
        x1 = np.cos(np.deg2rad(fov)) * nearby_thresh
        y2 = np.sin(np.deg2rad(fov+step)) * nearby_thresh 
        x2 = np.cos(np.deg2rad(fov+step)) * nearby_thresh 
        ax.plot([y1, y2], [x1, x2], linestyle="solid", color="grey")
        


    # Labels
    ax.set_xlabel('$y$ [m]')
    ax.set_ylabel('$x$ [m]')

    ax.invert_xaxis()

    plt.draw()


def get_colors(pred_obj, conf_thresh):
    color_table = [(255/255, 255/255, 0, 1),
                (200/255, 0, 200/255, 1),
                (102/255, 0, 102/255, 1),
                (152/255, 0, 152/255, 1),
                (0, 255/255, 255/255, 1),
                (0, 200/255, 200/255, 1),
                (255/255, 153/255, 255/255, 1),
                (255/255, 113/255, 255/255, 1),
                (153/255, 102/255, 51/255, 1),
                (183/255, 132/255, 81/255, 1),
                (100/255, 80/255, 20/255, 1),
                (255/255,153/155, 0, 1),
                (240/255,130/155, 0, 1),
                (200/255,100/155, 0, 1),
                (224/255, 224/255, 224/255, 1),
                (190/255, 190/255, 190/255, 1),
                (240/255, 240/255, 240/255, 1),
                (128/255, 128/255, 0, 1),
                (40/255, 40/255, 0, 1),
                (200/255, 200/255, 0, 1),
                (0 , 0 , 200, 1),
                (0 , 0 , 150, 1),
                (0 , 0 , 255, 1)]
    color_arr = []
    for i,q in enumerate(pred_obj):
        if q > conf_thresh:
            print(i%len(color_table))
            color = color_table[i%len(color_table)]
            color_arr.append([i, color])
    return color_arr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_to_weights = "training_results/run_MLP9/best.pth"

# path_to_video = "/home/marten/Uni/Semester_4/src/TestData/955_2.avi"
# path_to_imu = "/home/marten/Uni/Semester_4/src/TestData/furuno_955.txt"
path_to_video = "/home/marten/Uni/Semester_4/src/TestData/22_2.avi"
path_to_imu = "/home/marten/Uni/Semester_4/src/TestData/furuno_22.txt"
# path_to_video = "/home/marten/Uni/Semester_4/src/TestData/videos_from_training/1004_2.avi"
# path_to_imu = "/home/marten/Uni/Semester_4/src/TestData/videos_from_training/furuno_1004.txt"
# path_to_video = "../TestData/videos_from_training/19_2.avi"
# path_to_imu = "../TestData/videos_from_training/furuno_19.txt"

# General settings
conf_thresh = .9    # threshhold of objectness pred -> only queries with pred_conf >= conf_thresh will be visualized
resize_coeffs = [0.5, 0.5] # applied to image before inference, 0 -> x, 1 -> y

# Model settings:
hidden_dim = 256    # embedding dim
enc_layers = 6      # encoding layers
dec_layers = 6      # decoding layers
dim_feedforward = 2048  # dim of ff layers in transformer layers
dropout = 0.1
nheads = 8          # transformear heads
pre_norm = True     # apply norm pre or post tranformer layer
input_dim_gt = 2    # Amount of datapoints of a query object before being transformed to embedding
use_embeddings = False

# Init Model
backbone = init_backbone(1e-5, hidden_dim)
transformer = init_transformer(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm)
model = DETR(
    backbone,
    transformer,
    input_dim_gt=2,
    aux_loss=False,
    use_embeddings=use_embeddings,
)
model.to(device)
model.eval()

print("Loading Weights...")
checkpoint = torch.load(path_to_weights, map_location=device)
model.load_state_dict(checkpoint['model'], strict=True)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of params:', n_parameters)


if not os.path.exists(path_to_video) and os.path.exists(path_to_imu):
    raise ValueError(f"Given Paths to data not valid: {path_to_video}, {path_to_imu}")
    
postprocess = PostProcess()
buoyGTData = GetGeoData()
imu_data = getIMUData(path_to_imu)
ship_pose = [imu_data[0][3],imu_data[0][4],imu_data[0][2]]
buoys_on_tile = buoyGTData.getBuoyLocations(ship_pose[0], ship_pose[1]) 
plt, ax = init_plot(two_dim=True)

cap = cv2.VideoCapture(path_to_video)
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # preprocess data (create image & query tensor)
    imu_curr = imu_data[frame_id]
    ship_pose = [imu_curr[3],imu_curr[4],imu_curr[2]]
    if buoyGTData.checkForRefresh(*ship_pose[0:2]):
        buoys_on_tile = buoyGTData.getBuoyLocations(*ship_pose[0:2])
    buoys_filtered = filterBuoys(ship_pose, buoys_on_tile)
    queries = torch.tensor(createQueryData(ship_pose, buoys_filtered), dtype=torch.float32)[...,0:2]

    colors = []
    if queries.numel() > 0: # if no queries could be generated -> skip inference
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        # normalize query inputs (dist and angle)
        queries[..., 0] = queries[..., 0] / 1000
        queries[..., 1] = queries[..., 1] / 180

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0,0), fx=resize_coeffs[0], fy=resize_coeffs[1])
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255    # standardizes img data & converts to 3xHxW

        # add batch dims
        queries = queries.unsqueeze(0).to(device)
        queries_mask = torch.full((1,queries.size(dim=1)), fill_value=True).to(device)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img, queries, queries_mask)
        outputs = postprocess(outputs, target_size=[frame.shape[0], frame.shape[1]])  # convert boxes from cxcyhw -> xyxy w.r.t. original image size
        pred_obj = outputs['objectness'].cpu().detach()
        pred_boxes = outputs['boxes'][pred_obj>=conf_thresh].cpu().detach()    # filter boxes based on objectness thresh
        colors = get_colors(pred_obj, conf_thresh)
        draw_boxes(frame, pred_boxes, pred_obj[pred_obj>=conf_thresh], colors)     # draw bbs with conf as text

    cv2.imshow("Buoy Association Transformer", frame)
    if frame_id % 8 == 0:
        live_plotting_2d(plt, ax, queries, colors)
    frame_id += 1

    key = cv2.waitKey(1)
    # Press 'q' to exit the loop
    if key == ord('q'):
        break
    if key == 32:
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()

