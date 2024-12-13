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


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm=0.1, logger=None):
    model.train()
    criterion.train()

    loss_total = []
    loss_obj = []
    loss_boxL1 = []
    loss_giou = []

    with tqdm(data_loader, desc=str(f"Train - Epoch {epoch}").ljust(15), ncols=150) as pbar:
        for images, queries, labels, queries_mask, labels_mask, name in pbar:
            images = images.to(device)
            queries = queries.to(device)
            queries = queries[..., 1:]  # remove index from queries (only for debugging reasons)
            labels = labels.to(device)
            queries_mask = queries_mask.to(device)
            labels_mask = labels_mask.to(device)

            outputs = model(images, queries, queries_mask)
            loss_dict = criterion(outputs, labels, queries_mask, labels_mask)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())
            loss_total.append(losses.item())
            loss_obj.append(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if 'loss_bce' in k).item())
            loss_boxL1.append(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if 'loss_bbox' in k).item())
            loss_giou.append(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if 'loss_giou' in k).item())
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
    with tqdm(data_loader, desc=str(f"Val - Epoch {epoch}").ljust(15), ncols=150) as pbar:
        for images, queries, labels, queries_mask, labels_mask, name in pbar:
            images = images.to(device)
            queries = queries.to(device)
            queries = queries[..., 1:]  # remove index from queries (only for debugging reasons)
            labels = labels.to(device)
            queries_mask = queries_mask.to(device)
            labels_mask = labels_mask.to(device)

            outputs = model(images, queries, queries_mask)
            loss_dict = criterion(outputs, labels, queries_mask, labels_mask)
            weight_dict = criterion.weight_dict

            losses = losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())
            loss_total.append(losses.item())
            loss_obj.append(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if 'loss_bce' in k).item())
            loss_boxL1.append(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if 'loss_bbox' in k).item())
            loss_giou.append(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if 'loss_giou' in k).item())
            tqdm_str = {"Loss": f"{round(sum(loss_total)/len(loss_total) ,3)}",
                        "Loss Obj": f"{round(sum(loss_obj)/len(loss_obj), 3)}",
                        "Loss BoxL1": f"{round(sum(loss_boxL1)/len(loss_boxL1), 3)}",
                        "Loss Giou": f"{round(sum(loss_giou)/len(loss_giou),3)}"}
            pbar.set_postfix(tqdm_str)
            if logger is not None:
                logger.computeStats(outputs, labels.cpu().detach(), queries_mask.cpu().detach(), labels_mask.cpu().detach(), mode='val')

    if logger is not None:
        losses ={"loss_total": sum(loss_total)/len(loss_total), "loss_obj": sum(loss_obj)/len(loss_obj),
                "loss_boxL1": sum(loss_boxL1)/len(loss_boxL1), "loss_giou": sum(loss_giou)/len(loss_giou)}
        logger.updateLosses(losses, epoch, 'val')
        logger.printCF(thresh = 0.5, mode='val') # Print Confusion Matrix for threshold of 0.5
        print("state_dict")
        print(logger.stats_dict)
        return losses
    else:
        return None


###########
# Settings
###########

# general
transfer_learning = True    # Loads prev provided weights
load_optim_state = False    # Loads state of optimizer / training if set to True
start_epoch = 0             # set this if continuing prev training
path_to_weights = r"detr-r50-e632da11.pth" 
output_dir = "test"

# Backbone
lr_backbone = 1e-5

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
distributed = False

# Dataset
path_to_dataset = "/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset1/dataset.yaml"
if distributed:
    path_to_dataset = "/data/mkreis/dataset/dataset.yaml"

# Loss
aux_loss = True
bce_loss_coef = 1
bbox_loss_coef = 2
giou_loss_coef = 5

# Optimizer / DataLoader
lr = 1e-4
batch_size=8
if distributed:
    batch_size = 2*torch.cuda.device_count()
weight_decay=1e-3
epochs=80
lr_drop=200
clip_max_norm=0.0
num_workers = 4
if distributed:
    num_workers = int(torch.cuda.device_count() * 10)


# Init Model
backbone = init_backbone(lr_backbone, hidden_dim)
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
if distributed:
    print("Training on multiple GPUs!")
    print("Using ", torch.cuda.device_count(), " GPUs")
    print("Batch Size:", batch_size)
    model = torch.nn.parallel.DataParallel(model)
    model_without_ddp = model.module
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

# Init Loss
weight_dict = {'loss_bce': bce_loss_coef, 'loss_bbox': bbox_loss_coef}
weight_dict['loss_giou'] = giou_loss_coef
if aux_loss:
    aux_weight_dict = {}
    for i in range(dec_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
losses = ['labels', 'boxes']
criterion = SetCriterion(weight_dict, losses)

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
dataset_train = BuoyDataset(yaml_file=path_to_dataset, mode='train')
dataset_val = BuoyDataset(yaml_file=path_to_dataset, mode='val')

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

data_loader_train = DataLoader(dataset_train, batch_size, sampler=sampler_train, collate_fn=collate_fn, num_workers=num_workers)
data_loader_val = DataLoader(dataset_val, batch_size, sampler=sampler_val, drop_last=False, collate_fn=collate_fn, num_workers=num_workers)


# load init weights if performing transfer learning
if transfer_learning:
    print("loading weights..")
    checkpoint = torch.load(path_to_weights, map_location='cpu')
    del checkpoint['model']['class_embed.weight']
    del checkpoint['model']['class_embed.bias']
    model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    if load_optim_state:
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1


logger = BasicLogger()
print("Start training")
start_time = time.time()
best_loss = math.inf
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
        logger.saveLogs(output_dir)
        if val_results["loss_total"] < best_loss:
            print("Saved new model as best.pht")
            print("\n")
            logger.plotPRCurve(path=output_dir, mode='val')
            best_loss = val_results["loss_total"]
            save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, os.path.join(output_dir, "best.pth"))


total_time = time.time() - start_time
hours = int(total_time // 3600)
minutes = int((total_time - hours*3600) // 60)
seconds = int((total_time - hours*3600 - 60*minutes))
print(f'Training time {hours:02}:{minutes:02}:{seconds:02}')
