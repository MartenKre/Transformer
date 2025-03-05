from collections import defaultdict
import torch
import os
import time
import math
import sys
from tqdm import tqdm
import torchmetrics
from pprint import pprint
import numpy as np

from datasets.buoy_dataset import BuoyDataset, collate_fn
from torch.utils.data import DataLoader
from util.box_ops import box_iou, box_cxcywh_to_xyxy
from util.misc import BasicLogger

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
    num_classes=80  # unused
    hidden_dim=256
    num_queries=300 # unused
    position_embed_type='sine'
    feat_channels=[256, 256, 256]
    feat_strides=[8, 16, 32]
    num_levels=3
    num_decoder_points=32    # default 4
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


def prepare_ap_data(outputs, labels, labels_mask):
    src_boxes = outputs['pred_boxes']
    src_logits = outputs['pred_logits']
    labels = labels[..., 1:]
    preds = []
    target = []
    batch_size=src_boxes.size(0)
    for i in range(0, batch_size):
        dct = {"boxes": src_boxes[i],
               "scores": src_logits[i],
               "labels": torch.zeros_like(src_logits[i], device=src_logits.device, dtype=int)}
        preds.append(dct)

        dct2 = {"boxes": labels[i][labels_mask[i]],
                "labels": torch.zeros((labels_mask[i].sum()), device=labels_mask.device, dtype=int)}
        target.append(dct2)

    return preds, target

def computeIOU(bb_pred, bb_label):
    bb_pred = box_cxcywh_to_xyxy(bb_pred)
    bb_label = box_cxcywh_to_xyxy(bb_label)
    iou = box_iou(bb_pred, bb_label)[0]
    return torch.diag(iou)

def compute_metrics(outputs, labels, queries_mask, labels_mask, results_dict, iou_thresh=0.5, conf_thresh=0.90):
    src_logits = outputs['pred_logits'].cpu().detach()     # only get logits, that have corresponding label
    conf_mask = torch.full(src_logits.shape, fill_value=False)
    conf_mask[src_logits>=conf_thresh] = True       # mask for conf logits that are greater than thresh
    bb_filtered = outputs['pred_boxes'][conf_mask & labels_mask].cpu().detach()
    labels_filtered = labels[conf_mask & labels_mask][...,1:]
    fp_conf = src_logits[conf_mask & ~labels_mask & queries_mask].numel()
    fn_conf = src_logits[~conf_mask & labels_mask].numel()  # fp through conf: has corresp label but is below conf level
    res = computeIOU(bb_filtered, labels_filtered)
    fp_iou = res[res<iou_thresh].numel()
    fn_iou = fp_iou
    tp = res.numel() - fp_iou

    results_dict['tp'] += tp
    results_dict['fp'] += fp_iou + fp_conf
    results_dict['fn'] += fn_iou + fn_conf
    results_dict['IoU'] += res.sum().item()
    results_dict['tp_match'] += bb_filtered.size(0)

def print_metrics(metrics):
    metrics["Precision"] = metrics["tp"] / (metrics["tp"]+ metrics["fp"])
    metrics["Recall"] = metrics["tp"] / (metrics["tp"]+ metrics["fn"])
    metrics["F1-Score"] = 2 * metrics["Precision"] * metrics["Recall"] / (metrics["Recall"]+ metrics["Precision"])
    metrics["Mean-IoU"] = metrics["IoU"] / metrics["tp_match"]
    for k,v in metrics.items():
        print(f"{k}:".ljust(6), v)

def print_latency(latency):
    latency = latency["time"] / latency["count"]
    print("Latency: ", round(latency), "ms")
    print("FPS: ", round(1 / (latency/1000), 2))

def compute_F1_over_dist(outputs, queries, labels, queries_mask, labels_mask, results_dict_distance, iou_thresh=0.5, conf_thresh=0.90):
    src_logits = outputs['pred_logits'].cpu().detach()     # only get logits, that have corresponding label
    bb_preds = outputs['pred_boxes'].cpu().detach()
    for b in range(0,src_logits.size(0)):    # batch size
        for q in range(0,src_logits.size(1)):    # query index
            if queries_mask[b, q] == True:
                dist = int(queries[b, q, 0] * 1000 // 50)
                if dist not in results_dict_distance:
                    results_dict_distance[dist] = {"tp": 0, "fp": 0, "fn": 0, "IoU": 0, "tp_match": 0}

                if src_logits[b,q] >= conf_thresh:
                    if labels_mask[b,q] == True:
                        res = computeIOU(bb_preds[b, q].unsqueeze(0), labels[b,q][...,1:].unsqueeze(0))
                        results_dict_distance[dist]["IoU"] += res
                        results_dict_distance[dist]["tp_match"] += 1
                        if res > iou_thresh:
                            results_dict_distance[dist]["tp"] += 1
                        else:
                            results_dict_distance[dist]["fp"] += 1
                            results_dict_distance[dist]["fn"] += 1
                    else:
                        results_dict_distance[dist]["fp"] += 1
                else:
                    if labels_mask[b,q] == True:
                        results_dict_distance[dist]["fn"] += 1

def save_F1_over_dist(metrics, test_dir):
    distances = sorted([k for k in metrics])
    f1_scores = []
    res = np.zeros((len(distances), 3))
    for i, k in enumerate(distances):
        p = metrics[k]["tp"] / (metrics[k]["tp"] + metrics[k]["fp"]) if metrics[k]["tp"]+ metrics[k]["fp"] > 0 else 0
        r = metrics[k]["tp"] / (metrics[k]["tp"] + metrics[k]["fn"]) if metrics[k]["tp"]+ metrics[k]["fn"] > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        iou = metrics[k]["IoU"] / metrics[k]["tp_match"] if metrics[k]["tp_match"] > 0 else 0
        res[i, 0] = k
        res[i, 1] = f1
        res[i, 2] = iou
        f1_scores.append(f1)
    print()
    print("F1 and IoU over distances:")
    print(res)
    np.save(os.path.join(test_dir, 'np_arr.npy'), res)


@torch.no_grad()
def test(model, data_loader, device, logger=None):
    model.eval()
    # ap_metric=torchmetrics.detection.MeanAveragePrecision(box_format="cxcywh", iou_type='bbox')

    metrics_dict = defaultdict(float)
    metrics_dict_distance = {}
    latency_dict = defaultdict(float)
    with tqdm(data_loader, desc=str(f"Test").ljust(8), ncols=150) as pbar:
        for images, queries, labels, queries_mask, labels_mask, name in pbar:
            start_time = time.perf_counter()

            images = images.to(device)
            queries = queries.to(device)
            queries = queries[..., 1:]  # remove index from queries (only for debugging reasons)
            labels = labels.to(device)
            queries_mask = queries_mask.to(device)
            labels_mask = labels_mask.to(device)

            outputs = model(images, query=queries, query_mask=queries_mask)

            latency_dict["time"] += (time.perf_counter() - start_time) * 1000
            latency_dict["count"] += images.size(0)

            # preds, target = prepare_ap_data(outputs, labels, labels_mask)
            # ap_metric.update(preds, target)
            compute_metrics(outputs, labels.cpu().detach(), queries_mask.cpu().detach(), labels_mask.cpu().detach(), metrics_dict)
            compute_F1_over_dist(outputs, queries.cpu().detach(), labels.cpu().detach(), queries_mask.cpu().detach(), labels_mask.cpu().detach(), metrics_dict_distance)
            if logger is not None:
                pass
                #logger.computeStats(outputs, labels.cpu().detach(), queries_mask.cpu().detach(), labels_mask.cpu().detach(), mode='val')

    print("Results:")
    print_metrics(metrics_dict)
    print()
    print_latency(latency_dict)
    print()
    save_F1_over_dist(metrics_dict_distance, output_dir)

    if logger is not None:
        logger.printCF(thresh = 0.5, mode='val')    # Print Confusion Matrix for threshold of 0.5
        ap50 = logger.print_mAP50(mode='val')
        logger.print_mAP50_95(mode="val")
        # res = ap_metric.compute()
        # pprint(res)
        return ap50
    else:
        return None


###########
# Settings
###########

# general
path_to_weights = r"training_results/rt-detr/run5/best_ap.pth" 
output_dir = "test_results"

# Transformer
hidden_dim = 256    # embedding dim
enc_layers = 6      # encoding layers
dec_layers = 6      # decoding layers
dim_feedforward = 2048  # dim of ff layers in transformer layers
dropout = 0.1
nheads = 8          # transformear heads
pre_norm = True     # apply norm pre or post tranformer layer
input_dim_gt = 2    # Amount of datapoints of a query object before being transformed to embedding
use_embeddings = False

# Multi GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
# path_to_dataset = "/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset2/dataset.yaml"
path_to_dataset = "/home/marten/Uni/Semester_4/src/TestData/TestLabeled/Generated_Sets/Transformer/dataset.yaml"

# Loss
aux_loss = False

# Optimizer / DataLoader
batch_size=1
weight_decay=1e-3
epochs=120
lr_drop=65
clip_max_norm=0.0
num_workers = 4


# Init Model
backbone = init_backbone()
encoder = init_hybrid_encoder()
decoder = init_decoder(True, 6)
model = init_rt_detr(backbone, encoder, decoder)
model.to(device)
model.eval()

model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

# Dataset
dataset_test = BuoyDataset(yaml_file=path_to_dataset, mode='test')

sampler_test = torch.utils.data.SequentialSampler(dataset_test)

data_loader_val = DataLoader(dataset_test, batch_size, sampler=sampler_test, drop_last=False, collate_fn=collate_fn, num_workers=num_workers)


# load init weights if performing transfer learning
print("loading weights..")
checkpoint = torch.load(path_to_weights, map_location='cpu')
model_without_ddp.load_state_dict(checkpoint['model'], strict=True)


logger = BasicLogger()
print("Start testing")
if not os.path.isdir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

logger.resetStats() # clear logger for new epoch

# validation
val_results = test(model, data_loader_val, device, logger)

logger.saveStatsLogs(output_dir, 0)
logger.plotPRCurve(path=output_dir, mode='val')
logger.plotConfusionMat(path=output_dir, thresh=0.5, mode='val')
logger.plotPRCurveDet(path=output_dir, mode="val")

logger.writeEpochStatsLog(path=output_dir, best_epoch=0)

print(f"Done! Test results saved to: {output_dir}")
