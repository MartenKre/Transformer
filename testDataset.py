from torch.utils.data import DataLoader 
from datasets.buoy_dataset import BuoyDataset, collate_fn
from torchvision import transforms
import numpy as np

train_dataset = BuoyDataset(yaml_file="/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset1/dataset.yaml", mode='train')
dataloader = DataLoader(train_dataset, batch_size = 4, shuffle=False, collate_fn=collate_fn)

for img, queries, labels in dataloader:
    print(img.shape())
    print(labels.shape())
    print(queries.shape())
