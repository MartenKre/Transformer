V2W-Transformer
========
This repo contains the modified architecture of DETR based Object Detection Models. It Replaces the fixed amount of object queries with a varying amount of sampled nearby chart markers. The spatial position of the chart markers w.r.t. the camera is encoded with an MLP and passed as an embedding to the decoder. After the final decoder layer, each chart marker embedding predicts its visibility for the given frame as well as bounding box coordinates.

![transformer_architecture](https://github.com/user-attachments/assets/8ffa77b7-5a87-48d3-849e-db56a00b888e)

## Dataset
The dataset has to be provided in the following format:
```
|
├── train
│   ├── images
│   |   ├── 00001.png
│   |   └── 00002.png
│   ├── labels
│   |   ├── 00001.txt
│   |   └── 00002.txt
│   └── queries
│       ├── 00001.txt
│       └── 00002.txt
├── test
├── val
└── dataset.yaml
```
The dataset.yaml contains the paths to the train test and val folders. Each folder consists of an image, label and query subfolder, where the ids of the individual samples can be used as cross reference to the other folders. 

A labels.txt file contains the bounding box annotations in YOLO format alongside the corresponding query id (first position). A query.txt file contains all sampled chart markers (queries) for the given frame, containing dist, bearing, lat, lng as well as its id.


## Training
To train the model, run:
```bash
python training.py
```
Hyperparameters as well as paths to weights and Datasets can be set under the SETTINGS section in the script.
The script also support multi GPU training (however not distributed on different compute nodes). This can be enabled by setting distributed to True.


## Testing
To test the model on labeled data run:
```bash
python test.py
```

## Inference on Video
To run inference on a video to compute the associations between chart markers and detected objects, run:
```bash
python buoyAssociation.py
```
Specify the Path to the Video and IMU File in the Function call (at the bottom of the script)
