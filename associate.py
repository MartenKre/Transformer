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

from models.rtdetr_postprocessor import PostProcess
from models.hybrid_encoder import HybridEncoder
from models.presnet import PResNet
from models.rtdetr_decoder import RTDETRTransformer
from models.rtdetr import RTDETR
from util.association_utility import getIMUData, filterBuoys, createQueryData, GetGeoData, LatLng2ECEF, T_ECEF_Ship
from util.plot_utils import plot_one_box


def init_hybrid_encoder():
    in_channels=[512, 1024, 2048]
    feat_strides=[8, 16, 32]
    hidden_dim=256
    nhead=8
    dim_feedforward = 1024
    dropout=0.0
    enc_act='gelu'
    use_encoder_idx=[2]
    num_encoder_layers=1
    pe_temperature=10000
    expansion=1.0
    depth_mult=1.0
    act='silu'
    eval_spatial_size=None
    return HybridEncoder(in_channels, feat_strides, hidden_dim, nhead, dim_feedforward, dropout, 
                         enc_act, use_encoder_idx, num_encoder_layers, pe_temperature, expansion,
                         depth_mult, act, eval_spatial_size)


def init_backbone():
    depth=50
    variant='d'
    num_stages=4
    return_idx=[1, 2, 3]
    act='relu'
    freeze_at=0
    freeze_norm=True
    pretrained=True
    return PResNet(depth, variant, num_stages, return_idx, act, freeze_at, freeze_norm, pretrained)


def init_decoder(aux_loss, dec_layers):
    num_classes=1  
    hidden_dim=256
    num_queries=100 
    position_embed_type='sine'
    feat_channels=[256, 256, 256]
    feat_strides=[8, 16, 32]
    num_levels=3
    num_decoder_points=4    # default 4
    nhead=8
    num_decoder_layers=dec_layers
    dim_feedforward=1024
    dropout=0.
    activation="relu"
    num_denoising=0
    label_noise_ratio=0.5
    box_noise_scale=1.0
    learnt_init_query=False     # set to False
    eval_spatial_size=None
    eval_idx=-1
    eps=1e-2
    aux_loss=aux_loss
    return RTDETRTransformer(num_classes, hidden_dim, num_queries, position_embed_type, feat_channels, feat_strides,
                             num_levels, num_decoder_points, nhead, num_decoder_layers, dim_feedforward, dropout,
                             activation, num_denoising, label_noise_ratio, box_noise_scale, learnt_init_query,
                             eval_spatial_size, eval_idx, eps, aux_loss)

def init_rt_detr(backbone, encoder, decoder):
    backbone=backbone
    encoder=encoder
    decoder=decoder
    multi_scale=None
    return RTDETR(backbone, encoder, decoder, multi_scale)


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


def init_plot():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
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
    ax.view_init(elev=60, azim=-180)  # Set view to make it look like a 2D XY plane

    plt.draw()

def get_colors(pred_obj, conf_thresh):
    color_table = [(255/255, 255/255, 0, 1),
                (102/255, 0, 102/255, 1),
                (0, 255/255, 255/255, 1),
                (255/255, 153/255, 255/255, 1),
                (153/255, 102/255, 51/255, 1),
                (255/255,153/155, 0, 1),
                (224/255, 224/255, 224/255, 1),
                (128/255, 128/255, 0, 1)]
    color_arr = []
    for i,q in enumerate(pred_obj):
        if q > conf_thresh:
            color = color_table[i%len(color_table)]
            color_arr.append([i, color])
    return color_arr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_to_weights = "training_results/rt_detr_obj_only/run1/best.pth"

path_to_video = "/home/marten/Uni/Semester_4/src/TestData/955_2.avi"
path_to_imu = "/home/marten/Uni/Semester_4/src/TestData/furuno_955.txt"
# path_to_video = "/home/marten/Uni/Semester_4/src/TestData/22_2.avi"
# path_to_imu = "/home/marten/Uni/Semester_4/src/TestData/furuno_22.txt"
# path_to_video = "/home/marten/Uni/Semester_4/src/TestData/videos_from_training/1004_2.avi"
# path_to_imu = "/home/marten/Uni/Semester_4/src/TestData/videos_from_training/furuno_1004.txt"
# path_to_video = "../TestData/videos_from_training/19_2.avi"
# path_to_imu = "../TestData/videos_from_training/furuno_19.txt"

# General settings
conf_thresh = .8    # threshhold of objectness pred -> only queries with pred_conf >= conf_thresh will be visualized
resize_coeffs = [0.5, 0.5] # applied to image before inference, 0 -> x, 1 -> y

# Init Model
backbone = init_backbone()
encoder = init_hybrid_encoder()
decoder = init_decoder(True, 6)
model = init_rt_detr(backbone, encoder, decoder)
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
plt, ax = init_plot()

cap = cv2.VideoCapture(path_to_video)
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # preprocess data (create image & query tensor)
    # imu_curr = imu_data[frame_id]
    # ship_pose = [imu_curr[3],imu_curr[4],imu_curr[2]]
    # if buoyGTData.checkForRefresh(*ship_pose[0:2]):
    #     buoys_on_tile = buoyGTData.getBuoyLocations(*ship_pose[0:2])
    # buoys_filtered = filterBuoys(ship_pose, buoys_on_tile)
    # queries = torch.tensor(createQueryData(ship_pose, buoys_filtered), dtype=torch.float32)[...,0:2]

    colors = []
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (0,0), fx=resize_coeffs[0], fy=resize_coeffs[1])
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255    # standardizes img data & converts to 3xHxW

    # add batch dims
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img, targets=None, query=None, query_mask=None)
    outputs = postprocess(outputs, target_size=[frame.shape[0], frame.shape[1]])  # convert boxes from cxcyhw -> xyxy w.r.t. original image size
    pred_obj = outputs['objectness'].cpu()
    pred_boxes = outputs['boxes'][pred_obj>=conf_thresh].cpu()    # filter boxes based on objectness thresh
    colors = get_colors(pred_obj, conf_thresh)
    draw_boxes(frame, pred_boxes, pred_obj[pred_obj>=conf_thresh], colors)     # draw bbs with conf as text

    cv2.imshow("Buoy Association Transformer", frame)
    frame_id += 1

    key = cv2.waitKey(1)
    # Press 'q' to exit the loop
    if key == ord('q'):
        break
    if key == 32:
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()

