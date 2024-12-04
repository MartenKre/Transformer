from torch.utils.data import DataLoader
from datasets.buoy_dataset import BuoyDataset
from torchvision import transforms
import numpy as np

# nomralize images: x' = (x-mean) / std
transform = transforms.Compose([
    transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                            std=[0.229, 0.224, 0.225])
])

train_dataset = BuoyDataset(yaml_file="/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset1/dataset.yaml", mode='train', transform=transform)
dataloader = DataLoader(train_dataset, batch_size = 1, shuffle=False)

for i, data in enumerate(train_dataset):
    print(data[0])
    print("image:", np.shape(data[0]))
    print("queries:", np.shape(data[1]))
    print("labels:", np.shape(data[2]))
    break
