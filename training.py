import torch
import os
import time
import math
import sys
from tqdm import tqdm

from torch._prims_common import check_pin_memory

from datasets.buoy_dataset import BuoyDataset, collate_fn
from torch.utils.data import DataLoader, DistributedSampler, dataloader
from models.detr import DETR, SetCriterion, PostProcess
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine 
from util.misc import is_main_process, save_on_master, reduce_dict, BasicLogger


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


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm = 0.1):
    model.train()
    criterion.train()

    logger = BasicLogger(delimiter = "   ")
    loss_total = []
    loss_obj = []
    loss_boxL1 = []
    loss_giou = []

    with tqdm(data_loader, desc=f"Train - Epoch" + str(epoch), ncols=180) as pbar:
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
            loss_total.append(round(losses.item(),3))
            loss_obj.append(round(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k =='loss_bce' in k).item(),3,))
            loss_boxL1.append(round(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k =='loss_bbox' in k).item(),3))
            loss_giou.append(round(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k =='loss_giou' in k).item(),3))
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

            # print("Loss: ", losses)
            # gradients = []
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         gradients.append(param.grad.flatten())
            # print("Max Grad: ", torch.max(torch.abs(torch.cat(gradients))))
            # print("Max Norm: ", torch.norm(torch.cat(gradients)))
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Layer {name}: ", torch.max(torch.abs(param.grad)))


            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"Before Clipping: NaN/Inf in gradients of {name}")

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)

            #print(model.class_embed.weight.grad)
            optimizer.step()

    return {"loss_total": sum(loss_total)/len(loss_total), "loss_obj": sum(loss_obj)/len(loss_obj),
            "loss_boxL1": sum(loss_boxL1/len(loss_boxL1)), "loss_giou": sum(loss_giou)/len(loss_giou)}
    

@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    loss_total = []
    loss_obj = []
    loss_boxL1 = []
    loss_giou = []
    with tqdm(data_loader, desc=f"Val", ncols=180) as pbar:
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
            loss_obj.append(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k =='loss_bce' in k).item())
            loss_boxL1.append(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k =='loss_bbox' in k).item())
            loss_giou.append(sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k =='loss_giou' in k).item())
            tqdm_str = {"Loss": f"{round(sum(loss_total)/len(loss_total) ,3)}",
                        "Loss Obj": f"{round(sum(loss_obj)/len(loss_obj), 3)}",
                        "Loss BoxL1": f"{round(sum(loss_boxL1)/len(loss_boxL1), 3)}",
                        "Loss Giou": f"{round(sum(loss_giou)/len(loss_giou),3)}"}
            pbar.set_postfix(tqdm_str)


    return {"loss_total": sum(loss_total)/len(loss_total), "loss_obj": sum(loss_obj)/len(loss_obj),
            "loss_boxL1": sum(loss_boxL1/len(loss_boxL1)), "loss_giou": sum(loss_giou)/len(loss_giou)}

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

# Multi GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
distributed = False

# Loss
aux_loss = True
bbox_loss_coef = 1
giou_loss_coef = 0.5

# Optimizer / DataLoader
lr = 1e-4
batch_size=1
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
    aux_loss=aux_loss,
)
model.to(device)

model_without_ddp = model
if distributed:
    print("Distributed Training!")
    print("Using ", torch.cuda.device_count(), " GPUs")
    model = torch.nn.parallel.DataParallel(model)
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

# if distributed:
#     sampler_train = DistributedSampler(dataset_train, shuffle=True)
#     sampler_val = DistributedSampler(dataset_val, shuffle=False)
# else:
#     sampler_train = torch.utils.data.RandomSampler(dataset_train)
#     sampler_val = torch.utils.data.SequentialSampler(dataset_val)

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


print("Start training")
start_time = time.time()
best_loss = math.inf
for epoch in range(start_epoch, epochs):
    if distributed:
        sampler_train.set_epoch(epoch)
    train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, clip_max_norm)
    lr_scheduler.step()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_paths = [os.path.join(output_dir, 'checkpoint.pth')]
        # extra checkpoint before LR drop and every 100 epochs
        if (epoch + 1) % lr_drop == 0 or (epoch + 1) % 100 == 0:
            checkpoint_paths.append(os.path.join(output_dir, f'checkpoint{epoch:04}.pth'))
        for checkpoint_path in checkpoint_paths:
            save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)

    val_stats = evaluate(model, criterion, data_loader_val, device)
    if output_dir:
        if val_stats["loss_total"] < best_loss:
            best_loss = val_stats["loss_total"]
            save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, os.path.join(output_dir, "best.pth"))

total_time = time.time() - start_time
total_time_str = str(time.datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
