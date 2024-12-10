"""Script to create Dataset in Yolo Format for Transformer (inlcuding images, buoyGT data, and BB Labels"""

import os
import json
from os.path import isdir
import numpy as np
import shutil
import cv2
from util.association_utility import GetGeoData, getIMUData, filterBuoys, createQueryData, haversineDist

def labelsJSON2Yolo(labels, queries):
    # fuction converts json data to yolo format with corresponding query ID
    result = []
    for BB in labels[1]["objects"]:
        if "distanceGT" in BB["attributes"]:
            # get query ID
            lat_BB = BB["attributes"]["buoyLat"]
            lng_BB = BB["attributes"]["buoyLng"]
            distances = list(map(lambda x: haversineDist(lat_BB, lng_BB, x[2], x[3]), queries))
            queryID = np.argmin(distances)
            if distances[queryID] > 30:
                if verbose:
                    print("\t \t Warning: No distance between query Buoy and Label buoy exceeds thresh")
                    print(lat_BB, lng_BB, distances)
                return None
           
            # get BB info in yolo format
            x1 = BB["x1"]
            y1 = BB["y1"]
            x2 = BB["x2"]
            y2 = BB["y2"]
            bbCenterX = ((x1 + x2) / 2) / 1920 
            bbCenterY = ((y1 + y2) / 2) / 1080 
            bbWidth = (x2-x1) / 1920 
            bbHeight = (y2-y1) / 1080 
            bbInfo = str(queryID) + " " + str(bbCenterX) + " " + str(bbCenterY) + " " + str(bbWidth) + " " + str(bbHeight) + "\n"

            result.append(bbInfo)

    return result 

# Settings:
mode = 'test' # train or test
verbose = False
resize_imgs = True
resize_coeffs = [0.5, 0.5]
target_dir = "/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset1"
data_path = "/home/marten/Uni/Semester_4/src/Trainingdata/labeled/"
os.makedirs(target_dir, exist_ok=True)

buoyGTData = GetGeoData()

# train data
if mode == 'train':
    target_dir = os.path.join(target_dir, "train")
if mode == 'test':
    target_dir = os.path.join(target_dir, "test")
os.makedirs(target_dir, exist_ok=True)
os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "queries"), exist_ok=True)
sample_counter = 0

datafolders = []
for folder in os.listdir(data_path):
    if os.path.isdir(os.path.join(data_path, folder)):
        if mode == 'train' and folder != 'Testdata' and folder != 'True_Negatives' and folder != "Boston_oak":
            datafolders.append(folder)
        elif mode == 'test' and folder == 'Testdata':
            datafolders.append(folder)

print("Folders Found: ", sorted(datafolders))

for folder in datafolders:
    parent_folder = os.path.join(data_path, folder)
    for subfolder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, subfolder)
        print("Processing: " , folder_path)
        images = os.path.join(folder_path, "images")
        imu = os.listdir(os.path.join(folder_path, "imu"))[0]
        labels = os.path.join(folder_path, "labels")
        imu_data = getIMUData(os.path.join(folder_path, "imu", imu)) 

        for sample in os.listdir(images):
            # copy image
            src_path_img = os.path.join(images, sample)
            sample_name = "0" * (5-len(list(str(sample_counter)))) + str(sample_counter)
            dest_path = os.path.join(target_dir, "images", sample_name + '.png')

            # create query file
            frame_id = int(sample.split(".")[0]) - 1
            imu_curr = imu_data[frame_id] 
            ship_pose = [imu_curr[3],imu_curr[4],imu_curr[2]]
            buoys_on_tile = buoyGTData.getBuoyLocations(ship_pose[0], ship_pose[1]) 
            filteredGT = filterBuoys(ship_pose, buoys_on_tile)
            queries = createQueryData(ship_pose, filteredGT)
            if len(queries) == 0:
                if verbose:
                    print(f"\t \t Skipping: No nearby bouys found for file {src_path_img}")
                continue
            queryFile = os.path.join(target_dir, "queries", sample_name + '.txt')

            # create labels file
            src_path = os.path.join(labels, sample+".json")
            label_data = json.load(open(src_path, 'r'))
            txtlabels = labelsJSON2Yolo(label_data, queries)
            if txtlabels is None:
                continue
            if len(txtlabels) == 0:
                if verbose:
                    print(f"\t \t skipping empty labels file: {src_path}")
                continue
            labelfile = os.path.join(target_dir, "labels", sample_name + '.txt')


            # save data
            if resize_imgs == False:
                shutil.copy(src_path_img, dest_path)
            else:
                img = cv2.imread(src_path_img)
                img_resized = cv2.resize(img, (0,0), fx=resize_coeffs[0], fy=resize_coeffs[1])
                cv2.imwrite(dest_path, img_resized)
            
            with open(queryFile, 'w') as f:
                data = [str(i) + " " + str(data[0]) + " " + str(data[1]) + " " + str(data[2]) + " " + str(data[3]) + "\n" for i,data in enumerate(queries)]
                f.writelines(data)

            with open(labelfile, 'w') as f:
                f.writelines(txtlabels)
            sample_counter += 1


print("DONE!")
print("Total Processed: ", datafolders)
