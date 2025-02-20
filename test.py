import torch
import os
import time
import math
import sys
from tqdm import tqdm

from datasets.buoy_dataset import BuoyDataset, collate_fn
from torch.utils.data import DataLoader
from models.detr import DETR, SetCriterion
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine 
from util.misc import save_on_master, BasicLogger


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


@torch.no_grad()
def test(model, data_loader, device, logger=None):
    model.eval()

    with tqdm(data_loader, desc=str(f"Test").ljust(8), ncols=150) as pbar:
        for images, queries, labels, queries_mask, labels_mask, name in pbar:
            images = images.to(device)
            queries = queries.to(device)
            queries = queries[..., 1:]  # remove index from queries (only for debugging reasons)
            labels = labels.to(device)
            queries_mask = queries_mask.to(device)
            labels_mask = labels_mask.to(device)

            outputs = model(images, queries, queries_mask)

            if logger is not None:
                logger.computeStats(outputs, labels.cpu().detach(), queries_mask.cpu().detach(), labels_mask.cpu().detach(), mode='val')

    if logger is not None:
        logger.printCF(thresh = 0.5, mode='val')    # Print Confusion Matrix for threshold of 0.5
        ap50 = logger.print_mAP50(mode='val')
        logger.print_mAP50_95(mode="val")
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
path_to_dataset = "/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset2/dataset.yaml"

# Loss
aux_loss = False

# Optimizer / DataLoader
batch_size=4
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

# training
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
