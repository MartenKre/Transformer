"""
Script to perform buoy association on a video 
"""
import torch
import os
import cv2
import random

from torchvision import transforms

from models.detr import DETR, PostProcess
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine 
from util.association_utility import getIMUData, filterBuoys, createQueryData, GetGeoData
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

def draw_boxes(frame, boxes, confs):
    colors = []
    for bb,conf in zip(boxes,confs):
        color = [random.randint(0, 255) for _ in range(3)]
        colors.append(color)
        txt = str(round(conf.item(), 3))
        plot_one_box(bb, frame, color=color, label=txt)
    return colors



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path_to_weights = "/home/marten/Uni/Semester_4/src/Transformer/run1/best_overall.pth"
path_to_video = "/home/marten/Uni/Semester_4/src/TestData/954_2.avi"
path_to_imu = "/home/marten/Uni/Semester_4/src/TestData/furuno_954.txt"
# path_to_video = "/home/marten/Uni/Semester_4/src/TestData/22_2.avi"
# path_to_imu = "/home/marten/Uni/Semester_4/src/TestData/furuno_22.txt"

# General settings
conf_thresh = 0.5    # threshhold of objectness pred -> only queries with pred_conf >= conf_thresh will be visualized
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

# Init Model
backbone = init_backbone(1e-5, hidden_dim)
transformer = init_transformer(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm)
model = DETR(
    backbone,
    transformer,
    input_dim_gt=2,
    aux_loss=False,
)
model.to(device)
model.eval()

print("Loading Weights...")
checkpoint = torch.load(path_to_weights, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of params:', n_parameters)


if not os.path.exists(path_to_video) and os.path.exists(path_to_imu):
    raise ValueError(f"Given Paths to data not valid: {path_to_video}, {path_to_imu}")
    
postprocess = PostProcess()
buoyGTData = GetGeoData()
imu_data = getIMUData(path_to_imu)
ship_pose = [imu_data[0][3],imu_data[0][4],imu_data[0][2]]
buoys_on_tile = buoyGTData.getBuoyLocations(ship_pose[0], ship_pose[1]) 

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
    buoys_filtered = filterBuoys(ship_pose, buoys_on_tile, fov_with_padding=110, dist_thresh=1000, nearby_thresh=30)
    queries = torch.tensor(createQueryData(ship_pose, buoys_filtered))[...,0:2]


    if queries.numel() > 0: # if no queries could be generated -> skip inference
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        # normalize query inputs (dist and angle)
        queries[..., 0] = queries[..., 0] / 1000
        queries[..., 1] = queries[..., 1] / 180

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0,0), fx=resize_coeffs[0], fy=resize_coeffs[1])
        img = torch.tensor(img).permute(2, 0, 1) / 255    # standardizes img data & converts to 3xHxW

        # add batch dims
        queries = queries.unsqueeze(0).to(device)
        queries_mask = torch.full((1,queries.size(dim=1)), fill_value=True).to(device)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img, queries, queries_mask)
        outputs = postprocess(outputs, target_size=[frame.shape[0], frame.shape[1]])  # convert boxes from cxcyhw -> xyxy w.r.t. original image size
        pred_obj = outputs['objectness'].cpu().detach()
        pred_boxes = outputs['boxes'][pred_obj>=conf_thresh].cpu().detach()    # filter boxes based on objectness thresh

        colors = draw_boxes(frame, pred_boxes, pred_obj[pred_obj>=conf_thresh])     # draw bbs with conf as text

    cv2.imshow("Buoy Association Transformer", frame)
    frame_id += 1

    key = cv2.waitKey(1)
    # Press 'q' to exit the loop
    if key == ord('q'):
        break

