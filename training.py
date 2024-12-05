from datasets.buoy_dataset import BuoyDataset, collate_fn
from torch.utils.data import DataLoader, DistributedSampler
from models.detr import DETR, SetCriterion
from models.transformer import Transformer
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine 
import torch


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

def train_one_epoch():
    pass

###########
# Settings
###########

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
weight_dict = {'loss_ce': 1, 'loss_bbox': bbox_loss_coef}
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
dataset_train = BuoyDataset(yaml_file="/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset1/dataset.yaml", mode='train')
dataset_val = BuoyDataset(yaml_file="/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset1/dataset.yaml", mode='val')

if distributed:
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, batch_size, drop_last=True)

data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=collate_fn, num_workers=num_workers)
data_loader_val = DataLoader(dataset_val, batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=collate_fn, num_workers=num_workers)


model.to(device)
model.train()
criterion.train()
train_one_epoch()
for img, queries, labels in dataloader:
    print(img.shape)
    print(queries.shape)
    print(labels.shape)
