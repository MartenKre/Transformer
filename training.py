import torch
import os
import time
import json
import math
import sys

from datasets.buoy_dataset import BuoyDataset, collate_fn
from torch.utils.data import DataLoader, DistributedSampler
from models.detr import DETR, SetCriterion, PostProcess
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine 
from util.misc import is_main_process, save_on_master, reduce_dict


def init_position_encoding(hidden_dim):
    N_steps = hidden_dim // 2
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    return position_embedding


def init_backbone(lr_backbone, hidden_dim, masks=False, backbone='resnet50', dilation=True):
    # masks are only used for image segmentation

    position_embedding = init_position_encoding(hidden_dim)
    train_backbone = lr_backbone > 0
    return_interm_layers = masks
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


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm = 0):

    model.train()
    criterion.train()

    for images, queries, labels, queries_mask, labels_mask in data_loader:
        images = images.to(device)
        queries = queries.to(device)
        labels = labels.to(device)
        queries_mask = queries_mask.to(device)
        labels_mask = labels_mask.to(device)

        outputs = model(images, queries, queries_mask)
        loss_dict = criterion(outputs, labels, queries_mask, labels_mask)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

###########
# Settings
###########

# general
transfer_learning = True    # Loads prev provided weights
load_optim_state = False    # Loads state of optimizer / training if set to True
start_epoch = 0             # set this if continuing prev training
path_to_weights = r"/home/marten/Uni/Semester_4/src/Transformer/detr-r50-e632da11.pth" 
output_dir = "run1"

# Backbone
lr_backbone = 1e-4

# Transformer
hidden_dim = 256    # embedding dim
enc_layers = 6      # encoding layers
dec_layers = 6      # decoding layers
dim_feedforward = 2048  # dim of ff layers in transformer layers
dropout = 0.1
nheads = 8          # transformear heads
pre_norm = True     # apply norm pre or post tranformer layer
input_dim_gt = 2    # Amount of datapoints of a query object before being transformed to embedding

# Multi GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cup')
distributed = False
gpu = ['0','1','2','3']

# Loss
aux_loss = True 
bbox_loss_coef = 2
giou_loss_coef = 5

# Optimizer / DataLoader
lr = 1e-4
lr_backbonet=1e-5
batch_size=2
weight_decay=1e-4
epochs=300
lr_drop=200
clip_max_norm=0.1
num_workers = 2


# Init Model
backbone = init_backbone(lr_backbone, hidden_dim)
transformer = init_transformer(hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers, pre_norm)
model = DETR(
    backbone,
    transformer,
    input_dim_gt=2,
    aux_loss=True,
)
model_without_ddp = model
if distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model_without_ddp = model.module
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

# Init Loss
weight_dict = {'loss_bce': 1, 'loss_bbox': bbox_loss_coef}
weight_dict['loss_giou'] = giou_loss_coef
if aux_loss:
    aux_weight_dict = {}
    for i in range(dec_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
losses = ['labels', 'boxes']
criterion = SetCriterion(weight_dict, losses)

# Init PostProcessor (only for evaluation)
postprocessors = {'bbox': PostProcess()}

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
dataset_train = BuoyDataset(yaml_file="/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset1/dataset.yaml", mode='train')
dataset_val = BuoyDataset(yaml_file="/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset1/dataset.yaml", mode='val')

if distributed:
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

#batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size, drop_last=True)
#data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=collate_fn, num_workers=num_workers)
data_loader_train = DataLoader(dataset_train, sampler=sampler_train, collate_fn=collate_fn, num_workers=num_workers)
data_loader_val = DataLoader(dataset_val, batch_size, sampler=sampler_val, drop_last=False, collate_fn=collate_fn, num_workers=num_workers)


# load init weights if performing transfer learning
if transfer_learning:
    checkpoint = torch.load(path_to_weights, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    if load_optim_state:
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

print("Start training")
start_time = time.time()
for epoch in range(start_epoch, epochs):
        if distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, clip_max_norm)
        lr_scheduler.step()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, device, output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if output_dir and is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            (output_dir / 'eval').mkdir(exist_ok=True)
            filenames = ['latest.pth']
            if epoch % 50 == 0:
                filenames.append(f'{epoch:03}.pth')
            for name in filenames:
                torch.save(coco_evaluator.coco_eval["bbox"].eval,
                            output_dir / "eval" / name)

total_time = time.time() - start_time
total_time_str = str(time.datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
