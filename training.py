from json import decoder, encoder
import torch
import os
import time
import math
import sys
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from datasets.buoy_dataset import BuoyDataset, collate_fn
from torch.utils.data import DataLoader
from models.hybrid_encoder import HybridEncoder
from models.rtdetr_decoder import RTDETRTransformer
from models.presnet import PResNet
from models.rtdetr import RTDETR
from models.rtdetr_criterion import SetCriterion
from util.misc import save_on_master, BasicLogger, prepare_ap_data
from models.rtdetr_criterion_obj_det import SetCriterion
from models.matcher import HungarianMatcher
from pprint import pprint
import torchmetrics

def init_obj_det_critetion():
    weight_dict_loss = {"loss_labels": 1, "loss_bbox": 5, "loss_giou": 2}
    losses = ["labels", "boxes"]
    weight_dict_matcher = {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}
    matcher = HungarianMatcher(weight_dict_matcher)
    criterion = SetCriterion(matcher, weight_dict_loss, losses, num_classes=1)
    return criterion

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
    num_queries=300 
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


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm=0.1, logger=None):
    model.train()
    criterion.train()

    loss_total = []
    loss_obj = []
    loss_boxL1 = []
    loss_giou = []

    with tqdm(data_loader, desc=str(f"Train - Epoch {epoch}").ljust(16), ncols=150) as pbar:
        for images, queries, labels, queries_mask, labels_mask, name, target in pbar:
            images = images.to(device)
            queries = queries.to(device)
            queries = queries[..., 1:]  # remove index from queries (only for debugging reasons)
            labels = labels.to(device)
            queries_mask = queries_mask.to(device)
            labels_mask = labels_mask.to(device)
            target = [{k: v.to(device) for k, v in dict_t.items()} for dict_t in target]

            outputs = model(images, targets=target, query=queries, query_mask=queries_mask)
            # loss_dict = criterion(outputs, labels, queries_mask, labels_mask)
            loss_dict = criterion(outputs, target)
            
            losses = sum(v for v in loss_dict.values())
            loss_total.append(losses.item())
            loss_obj.append(sum(loss_dict[k] for k in loss_dict.keys() if 'loss_labels' in k).item())
            loss_boxL1.append(sum(loss_dict[k] for k in loss_dict.keys() if 'loss_bbox' in k).item())
            loss_giou.append(sum(loss_dict[k] for k in loss_dict.keys() if 'loss_giou' in k).item())
            tqdm_str = {"Loss": f"{round(sum(loss_total)/len(loss_total) ,3)}",
                        "Loss Obj": f"{round(sum(loss_obj)/len(loss_obj), 3)}",
                        "Loss BoxL1": f"{round(sum(loss_boxL1)/len(loss_boxL1), 3)}",
                        "Loss Giou": f"{round(sum(loss_giou)/len(loss_giou),3)}"}
            pbar.set_postfix(tqdm_str)

            loss_value = losses.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)

            optimizer.step()

    if logger is not None:
        losses ={"loss_total": sum(loss_total)/len(loss_total), "loss_obj": sum(loss_obj)/len(loss_obj),
                "loss_boxL1": sum(loss_boxL1)/len(loss_boxL1), "loss_giou": sum(loss_giou)/len(loss_giou)}
        logger.updateLosses(losses, epoch, 'train')
        return losses
    else:
        return None


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch, logger=None):
    model.eval()
    criterion.eval()

    loss_total = []
    loss_obj = []
    loss_boxL1 = []
    loss_giou = []
    ap_metric = torchmetrics.detection.MeanAveragePrecision(box_format="cxcywh", iou_type="bbox")
    with tqdm(data_loader, desc=str(f"Val - Epoch {epoch}").ljust(16), ncols=150) as pbar:
        for images, queries, labels, queries_mask, labels_mask, name, target in pbar:
            images = images.to(device)
            queries = queries.to(device)
            queries = queries[..., 1:]  # remove index from queries (only for debugging reasons)
            labels = labels.to(device)
            queries_mask = queries_mask.to(device)
            labels_mask = labels_mask.to(device)
            target = [{k: v.to(device) for k, v in dict_t.items()} for dict_t in target]

            outputs = model(images, targets=target, query=queries, query_mask=queries_mask)
            # loss_dict = criterion(outputs, labels, queries_mask, labels_mask)
            loss_dict = criterion(outputs, target)

            losses = sum(v for v in loss_dict.values())
            loss_total.append(losses.item())
            loss_obj.append(sum(loss_dict[k] for k in loss_dict.keys() if 'loss_labels' in k).item())
            loss_boxL1.append(sum(loss_dict[k] for k in loss_dict.keys() if 'loss_bbox' in k).item())
            loss_giou.append(sum(loss_dict[k] for k in loss_dict.keys() if 'loss_giou' in k).item())
            tqdm_str = {"Loss": f"{round(sum(loss_total)/len(loss_total) ,3)}",
                        "Loss Obj": f"{round(sum(loss_obj)/len(loss_obj), 3)}",
                        "Loss BoxL1": f"{round(sum(loss_boxL1)/len(loss_boxL1), 3)}",
                        "Loss Giou": f"{round(sum(loss_giou)/len(loss_giou),3)}"}
            pbar.set_postfix(tqdm_str)

            preds, target = prepare_ap_data(outputs, labels, labels_mask)
            ap_metric.update(preds, target)
            if logger is not None:
                pass
                #logger.computeStats(outputs, labels.cpu().detach(), queries_mask.cpu().detach(), labels_mask.cpu().detach(), mode='val')

    if logger is not None:
        results = {"loss_total": sum(loss_total)/len(loss_total), "loss_obj": sum(loss_obj)/len(loss_obj),
                "loss_boxL1": sum(loss_boxL1)/len(loss_boxL1), "loss_giou": sum(loss_giou)/len(loss_giou)}
        logger.updateLosses(results, epoch, 'val')
        print()
        pprint(ap_metric.compute())
        # logger.printCF(thresh = 0.5, mode='val')    # Print Confusion Matrix for threshold of 0.5
        # ap50 = logger.print_mAP50(mode='val')
        # logger.print_mAP50_95(mode="val")
        # results['AP50'] = ap50
        return results
    else:
        return None


###########
# Settings
###########

# general
transfer_learning = True    # Loads prev provided weights
load_optim_state = False    # Loads state of optimizer / training if set to True
start_epoch = 0             # set this if continuing prev training
path_to_weights = r"rtdetr_r50vd_6x_coco_from_paddle.pth" 
output_dir = "test"

# Backbone
lr_backbone = 1e-5


# Multi GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
distributed = False

# Dataset
path_to_dataset = "/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset2/dataset.yaml"
if distributed:
    path_to_dataset = "/data/mkreis/dataset2/dataset.yaml"

# Loss
bce_loss_coef = 1
bbox_loss_coef = 2
giou_loss_coef = 5
aux_loss = True
dec_layers = 6

# Optimizer / DataLoader
lr = 2e-4
lr_backbone = 1e-5
batch_size=4
if distributed:
    batch_size = 8*torch.cuda.device_count()
weight_decay=1e-3
epochs=120
lr_drop=65
clip_max_norm=0.0
num_workers = 4
if distributed:
    num_workers = 60


# Init Model
backbone = init_backbone()
encoder = init_hybrid_encoder()
decoder = init_decoder(aux_loss, dec_layers)
model = init_rt_detr(backbone, encoder, decoder)
model.to(device)


model_without_ddp = model
if distributed:
    print("Training on multiple GPUs!")
    print("Using ", torch.cuda.device_count(), " GPUs")
    print("Batch Size:", batch_size)
    model = torch.nn.parallel.DataParallel(model)
    model_without_ddp = model.module
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

# Init Loss
# weight_dict = {'loss_bce': bce_loss_coef, 'loss_bbox': bbox_loss_coef}
# weight_dict['loss_giou'] = giou_loss_coef
# if aux_loss:
#     aux_weight_dict = {}
#     for i in range(dec_layers - 1):
#         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
#     weight_dict.update(aux_weight_dict)
# losses = ['labels', 'boxes']
# criterion = SetCriterion(weight_dict, losses)
criterion = init_obj_det_critetion()

# Init Optim
param_dicts = [
    {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": lr_backbone,
    },
]
optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_drop)

# Dataset
dataset_train = BuoyDataset(yaml_file=path_to_dataset, mode='train', augment=True)
dataset_val = BuoyDataset(yaml_file=path_to_dataset, mode='val')

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

data_loader_train = DataLoader(dataset_train, batch_size, sampler=sampler_train, collate_fn=collate_fn, num_workers=num_workers)
data_loader_val = DataLoader(dataset_val, batch_size, sampler=sampler_val, drop_last=False, collate_fn=collate_fn, num_workers=num_workers)


# load init weights if performing transfer learning
if transfer_learning:
    print("loading weights..")
    weights = torch.load(path_to_weights, map_location='cpu')
    checkpoint = weights if 'model' in weights else {'model': weights}
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    if load_optim_state:
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1


logger = BasicLogger()
print("Start training")
start_time = time.time()
best_ap = -1
best_epoch = -1
for epoch in range(start_epoch, epochs):
    logger.resetStats() # clear logger for new epoch

    # training
    train_results = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, clip_max_norm, logger)
    lr_scheduler.step()

    # validation
    val_results = evaluate(model, criterion, data_loader_val, device, epoch, logger)

    if output_dir:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        logger.saveLossLogs(output_dir)
        # logger.saveStatsLogs(output_dir, epoch)
        logger.plotLoss(output_dir)
        save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, os.path.join(output_dir, "best.pth"))
        # if val_results["AP50"] > best_ap:
        #     print("Saved new model as best.pht")
        #     logger.plotPRCurve(path=output_dir, mode='val')
        #     logger.plotConfusionMat(path=output_dir, thresh = 0.5, mode='val')
        #     logger.plotPRCurveDet(path=output_dir, mode="val")
        #     best_ap = val_results["AP50"]
        #     best_epoch = epoch
        #     save_on_master({
        #         'model': model_without_ddp.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #     }, os.path.join(output_dir, "best.pth"))


total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time - hours*3600) // 60)
seconds = int((total_time - hours*3600 - 60*minutes))
print(f'Training time {hours:02}:{minutes:02}:{seconds:02}')
logger.writeEpochStatsLog(path=output_dir, best_epoch=best_epoch)
print("Best Val results in epoch: ", best_epoch)
