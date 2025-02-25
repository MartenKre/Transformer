from collections import defaultdict
import torch
import os
import time
import math
import sys
from tqdm import tqdm
import torchmetrics
from pprint import pprint

from datasets.buoy_dataset import BuoyDataset, collate_fn
from torch.utils.data import DataLoader
from util.box_ops import box_iou, box_cxcywh_to_xyxy
from models.detr import DETR
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine 
from util.misc import BasicLogger


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

@torch.no_grad()
def test(model, data_loader, device, logger=None):
    model.eval()
    # ap_metric=torchmetrics.detection.MeanAveragePrecision(box_format="cxcywh", iou_type='bbox')

    metrics_dict = defaultdict(float)
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

            outputs = model(images, queries, queries_mask)

            latency_dict["time"] += (time.perf_counter() - start_time) * 1000
            latency_dict["count"] += images.size(0)

            # preds, target = prepare_ap_data(outputs, labels, labels_mask)
            # ap_metric.update(preds, target)
            compute_metrics(outputs, labels.cpu().detach(), queries_mask.cpu().detach(), labels_mask.cpu().detach(), metrics_dict)
            if logger is not None:
                logger.computeStats(outputs, labels.cpu().detach(), queries_mask.cpu().detach(), labels_mask.cpu().detach(), mode='val')

    print("Results:")
    print_metrics(metrics_dict)
    print()
    print_latency(latency_dict)
    print()

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
path_to_weights = r"training_results/run_MLP10_newMet/best.pth" 
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
backbone = init_backbone(lr_backbone=0.1, hidden_dim=hidden_dim)
transformer = init_transformer(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm)
model = DETR(
    backbone,
    transformer,
    input_dim_gt=2,
    aux_loss=aux_loss,
    use_embeddings=use_embeddings,
)
model.to(device)

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

logger.resetStats() # clear logger for new epoch

# validation
val_results = test(model, data_loader_val, device, logger)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
logger.saveStatsLogs(output_dir, 0)
logger.plotPRCurve(path=output_dir, mode='val')
logger.plotConfusionMat(path=output_dir, thresh=0.5, mode='val')
logger.plotPRCurveDet(path=output_dir, mode="val")

logger.writeEpochStatsLog(path=output_dir, best_epoch=0)

print(f"Done! Test results saved to: {output_dir}")
