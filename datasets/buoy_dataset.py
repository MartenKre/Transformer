"""Custom Dataset for Buoy Data"""
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import torch
import yaml
import os
import numpy as np
import cv2


def collate_fn(batch):
    img, queries, labels, queries_mask, labels_mask, name = zip(*batch)
    img = torch.stack(img, dim=0)
    pad_q = pad_sequence(queries, batch_first=True, padding_value = 0.0)
    pad_l = pad_sequence(labels, batch_first=True, padding_value = 0.0)
    pad_mask_q = pad_sequence(queries_mask, batch_first=True, padding_value=False)
    pad_mask_l = pad_sequence(labels_mask, batch_first=True, padding_value=False)

    
    return img, pad_q, pad_l, pad_mask_q, pad_mask_l, name
    

class BuoyDataset(Dataset):
    def __init__(self, yaml_file, mode='train', transform=True) -> None:
        # mode: train/test/val
        super().__init__()

        self.yaml_file = yaml_file
        if mode in ["train", "test", "val"]:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode ({mode}) for DataSet")
        self.data_path = None

        tf = transforms.Compose([
            transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                                    std=[0.229, 0.224, 0.225])
        ])
        self.transform = None
        if transform:
            self.transform = tf
        self.processYAML()

        self.labels = sorted(os.listdir(os.path.join(self.data_path, "labels")))
        self.images = sorted(os.listdir(os.path.join(self.data_path, "images")))
        self.queries = sorted(os.listdir(os.path.join(self.data_path, "queries")))

        self.checkdataset()

    def processYAML(self):
        if not os.path.exists(self.yaml_file):
            raise ValueError(f"Path to Dataset not found - Incorrect YAML File Path: {self.yaml_file}")
        with open(self.yaml_file, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            if self.mode in data:
                self.data_path = data[self.mode]
                if not os.path.exists(self.data_path):
                    raise ValueError(f"Incorrect path to {self.mode} folder in YAML file: {self.data_path}")
            else:
                raise ValueError(f"YAML file does not contain path to {self.mode} folder")

    def checkdataset(self):
        for label, image, query in zip(self.labels, self.images, self.queries):
            if not image.split(".")[0] == label.split('.')[0] == query.split('.')[0]:
                print(f"Warning: {label}, {image}, {query}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.data_path, "images", self.images[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = torch.tensor(np.loadtxt(os.path.join(self.data_path, 'labels', self.labels[index])), dtype=torch.float32)
        queries = torch.tensor(np.loadtxt(os.path.join(self.data_path, 'queries', self.queries[index])),
                               dtype=torch.float32)[..., 0:3] # only take the first two datapoints in the label file

        # ensure 2D shape:
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)

        # normalize query inputs (dist and angle)
        queries[..., 1] = queries[..., 1] / 1000
        queries[..., 2] = queries[..., 1] / torch.pi

        labels_extended = torch.zeros(queries.size(dim=0), 5, dtype=torch.float32)
        labels_extended[labels[:, 0].int(), :] = labels[:, :]

        labels_mask = torch.full((1, labels_extended.size(dim=0)), fill_value=False).squeeze(0)
        labels_mask[labels[:, 0].int()] = True

        queries_mask = torch.full((1,queries.size(dim=0)), fill_value=True).squeeze(0)

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2, 0, 1) / 255

        name = os.path.join(self.data_path, "images", self.images[index])

        sample = (img, queries, labels_extended, queries_mask, labels_mask, name)
        return sample
        
