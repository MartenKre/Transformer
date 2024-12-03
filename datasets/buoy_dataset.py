"""Custom Dataset for Buoy Data"""
from torch.utils.data import Dataset
import yaml
import os
import numpy as np
import cv2

class BuoyDataset(Dataset):
    def __init__(self, yaml_file, mode='train', transform=None) -> None:
        # mode: train/test/val
        super().__init__()

        self.yaml_file = yaml_file
        if mode in ["train", "test", "val"]:
            self.mode = mode
        else:
            raise ValueError(f"Invalid mode ({mode}) for DataSet")
        self.data_path = None
        self.transform = transform
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
                    raise ValueError(f"Incorrect path to {self.mode} folder in YAML file")
            else:
                raise ValueError(f"YAML file does not contain path to {self.mode} folder")

    def checkdataset(self):
        for label, image, query in zip(self.labels, self.images, self.queries):
            assert image.split(".")[0] == label.split('.')[0] == query.split('.')[0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.data_path, "images", self.images[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = np.loadtext(self.labels[index])
        queries = np.loadtext(self.queries[index])
        sample = (img, labels, queries)

        if self.transform:
            sample = self.transform(sample)

        return sample
        
