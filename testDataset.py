from datasets import buoy_dataset
from torch.utils.data import DataLoader

train_dataset = buoy_dataset(yaml_file="/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset1/dataset.yaml", mode='train')
dataloader = DataLoader(train_dataset, batch_size = 1, shuffle=False)

for i, data in enumerate(train_dataset):
    print(data)
    break
