from torch.utils.data import DataLoader 
from datasets.buoy_dataset import BuoyDataset, collate_fn

train_dataset = BuoyDataset(yaml_file="/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset1/dataset.yaml", mode='train')
dataloader = DataLoader(train_dataset, batch_size = 2, shuffle=False, collate_fn=collate_fn)

for img, queries, labels, q_mask, l_mask in dataloader:
    print("-------------")
    print("shapes:")
    print(img.shape)
    print(queries.shape)
    print(labels.shape)
    print(q_mask.shape)
    print(l_mask.shape)
    print("Qsample1:")
    print(queries[0])
    print(q_mask[0])
    print("Qsample2:")
    print(queries[1])
    print(q_mask[1])
    print("Lsample1:")
    print(labels[0])
    print(l_mask[0])
    print("Lsample2:")
    print(labels[1])
    print(l_mask[1])
