"""Script to create Dataset in Yolo Format for Transformer (inlcuding images, buoyGT data, and BB Labels"""

import os
import json
from os.path import exists
import geojson
import numpy as np
import pyproj
from math import sin, cos, asin, sqrt, radians, floor
import shutil

# class do get relevant buoy GT positions (Decoder Input)
class GetGeoData():
    def __init__(self, file="/home/marten/Uni/Semester_4/src/BuoyAssociation/utility/data/noaa_navigational_aids.geojson", tile_size = 0.1):
        self.file = file    # path to geojson file
        self.tile_size = tile_size     # size of tile for which buoys are returned (in degrees)
        try:
            f = open(file)
            self.data = geojson.load(f)
        except:
            raise ValueError(f"Cannot open Geojson File: {self.file}")
        self.tile_center = None

    def getBuoyLocations(self, pos_lat, pos_lng):
        # Function returns buoy info for all buoys within self.tile_size from given pos
        # Arguments: pos_lat & pos_lng are geographical coordinates
        self.tile_center = {"lat": pos_lat, "lng": pos_lng}
        buoys = []
        for buoy in self.data["features"]:
            buoy_lat = buoy["geometry"]["coordinates"][1]
            buoy_lng = buoy["geometry"]["coordinates"][0]
            if abs(buoy_lng - pos_lng) < self.tile_size and abs(buoy_lat - pos_lat) < self.tile_size:
                buoys.append(buoy)
        return buoys

def LatLng2ECEF(lat, lng, alt = 0):
    # function expects lat & lng to be in deg
    # function returns xyz of ECEF
    lat = np.radians(lat)
    lng = np.radians(lng)
    a = 6378137.0 # equatorial radius
    b = 6356752.0 # polar radius
    e_sq = 1 - (b**2)/(a**2) 
    f = 1 - b/a

    N = a**2 / (np.sqrt(a**2 * np.cos(lat)**2 + b**2 * np.sin(lat)**2))

    X = (N + alt) * np.cos(lat) * np.cos(lng)
    Y = (N + alt) * np.cos(lat) * np.sin(lng)
    Z = ((1 - f)**2 * N + alt) * np.sin(lat)
    return X, Y, Z

def getRotationMatrix(a, axis='z'):
    R = np.eye(3)
    if axis == 'x' or axis == 'X':
        R = np.array([[1, 0, 0],
                        [0, np.cos(a), -np.sin(a)],
                        [0, np.sin(a), np.cos(a)]])
    elif axis == 'y' or axis == 'Y':
        R = np.array([[np.cos(a), 0, np.sin(a)],
                        [0, 1, 0],
                        [-np.sin(a), 0 , np.cos(a)]])
    elif axis == 'z' or axis == 'Z':
        R = np.array([[np.cos(a), -np.sin(a), 0],
                        [np.sin(a), np.cos(a), 0],
                        [0, 0 ,1]])
    else:
        raise ValueError(f"axis {axis} is not a valid argument")
    return R

def ECEF2LatLng(x, y, z, rad=False):
    # function returns lat & lng coordinates in Deg
    transformer = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'})
    lon1, lat1, alt1 = transformer.transform(x,y,z,radians=rad)
    return lat1, lon1, alt1


def T_ECEF_ENU(x, y, z):
    # returns transformation matrix ECEF_T_ENU
    # ENU: z to top, y to north, x to east
    # expects lat & lng in radians
    T = np.eye(4)
    lat, lng, alt = ECEF2LatLng(x, y, z, rad=True)
    R = np.array([[-np.sin(lng), -np.cos(lng)*np.sin(lat), np.cos(lng) * np.cos(lat)],
                    [np.cos(lng), -np.sin(lng)*np.sin(lat), np.sin(lng)* np.cos(lat)],
                    [0, np.cos(lat), np.sin(lat)]])
    t = np.array([x, y, z])# translation vector
    T[:3,:3] = R
    T[:3,3] = t
    return T

def T_ENU_Ship(heading):
    # returns transformatin matrix ENU_T_ship
    # expects heading to be in deg
    # ENU: z top, y north, x east
    # Ship, x front, y left, z top
    T = np.eye(4)
    angle = np.radians(90 - heading)
    R = getRotationMatrix(angle, axis='z')    # rotate to correct heading around z
    T[:3,:3] = R
    return T


def T_ECEF_Ship(x, y, z, heading):
    # returns transformtion matrix ECEF_T_Ship
    # expects arguments xyz to be ship pos in ECEF and ships heading in deg
    ECEF_T_ENU = T_ECEF_ENU(x,y,z)
    ENU_T_Ship = T_ENU_Ship(heading)

    ECEF_T_SHIP = ECEF_T_ENU @ ENU_T_Ship
    return ECEF_T_SHIP

def haversineDist(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r * 1e3

def filterBuoys(ship_pose, buoyCoords, fov_with_padding=110, dist_thresh=1000, nearby_thresh=30):
    """function selects nearby gt buoys (relative to ship pos) from a list containing buoy Coordinates
    Args:
        ship_pose: list of form [lat,lng,heading]
        buoyCoords: list of form [[lat1,lng1], [lat2, lng2], ...]
        fov_with_padding: fov of camera plus additional padding to account for inaccuracies
        dist_thresh: only buoys inside this threshold will be considered
        nearby_thresh: all buoys inside this thresh will pe added, even if outside of fov
     """

    selected_buoys = []
    # compute transformation matrix from ecef to ship cs
    x, y, z = LatLng2ECEF(ship_pose[0], ship_pose[1])  # ship coords in ECEF
    Ship_T_ECEF = np.linalg.pinv(T_ECEF_Ship(x,y,z,ship_pose[2]))   # transformation matrix between ship and ecef

    for buoy in buoyCoords:
        lat = buoy["geometry"]["coordinates"][1]
        lng = buoy["geometry"]["coordinates"][0]
        x,y,z = LatLng2ECEF(lat, lng)
        pos_bouy = Ship_T_ECEF @ np.array([x,y,z,1]) # transform buoys from ecef to ship cs
        bearing = np.rad2deg(np.arctan2(pos_bouy[1],pos_bouy[0]))   # compute bearing of buoy
        dist_to_ship = haversineDist(lat, lng, ship_pose[0], ship_pose[1])  # compute dist to ship
        if abs(bearing) <= fov_with_padding / 2 and dist_to_ship <= dist_thresh:
            # include buoys that are within fov+padding and inside maxdist
            selected_buoys.append((lat, lng))
        elif dist_to_ship <= nearby_thresh:
            # also include nearby buoys not inside FOV
            selected_buoys.append((lat, lng))

    return list(set(selected_buoys))

def getIMUData(path):
    # function returns IMU data as list

    if os.path.isfile(path):
        result = []
        with open(path, 'r') as f:
            data = f.readlines()
            for line in data:
                content = line.split(",")
                line = [float(x) for x in content]
                result.append(line)
    else:
        files = os.listdir(path)
        filename = [f for f in files if f.endswith('.txt')][0]
        path = os.path.join(path, filename)
        result = []
        with open(path, 'r') as f:
            data = f.readlines()
            for line in data:
                content = line.split(",")
                line = [float(x) for x in content]
                result.append(line)
        if len(result) == 0:
            print("No IMU data found, check path: {path}")
    return result

def createQueryData(ship_pose, buoyList):
    # function computes query data (dist, bearing) based on current ship pose and buoyList
    
    result = []
    for buoy in buoyList:
        # dist
        dist = haversineDist(*ship_pose[:2],*buoy)

        # bearing
        x,y,z = LatLng2ECEF(*ship_pose[:2])
        ECEF_T_SHIP = T_ECEF_Ship(x,y,z,ship_pose[2])
        x,y,z = LatLng2ECEF(*buoy)
        p_b = np.linalg.pinv(ECEF_T_SHIP) @ np.array([x, y, z, 1])
        bearing = np.rad2deg(np.arctan2(p_b[1], p_b[0]))

        result.append([dist, bearing, *buoy])

    return result

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
                print("Warning: No distance between query Buoy and Label buoy exceeds thresh")
                print(lat_BB, lng_BB, distances)
           
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


target_dir = "/home/marten/Uni/Semester_4/src/Trainingdata/Generated_Sets/Transformer_Dataset1"
data_path = "/home/marten/Uni/Semester_4/src/Trainingdata/labeled/StPete_BuoysOnly/"
os.makedirs(target_dir, exist_ok=True)

buoyGTData = GetGeoData()

# train data
target_dir = os.path.join(target_dir, "train")
os.makedirs(target_dir, exist_ok=True)
os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "queries"), exist_ok=True)
sample_counter = 0

# TODO: Add entire training folder (not only stpeteBuoysonly)
# TODO Add test data

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    images = os.path.join(folder_path, "images")
    imu = os.listdir(os.path.join(folder_path, "imu"))[0]
    labels = os.path.join(folder_path, "labels")
    imu_data = getIMUData(os.path.join(folder_path, "imu", imu)) 

    print(f"Processing {folder}")

    for sample in os.listdir(images):
        # copy image
        src_path = os.path.join(images, sample)
        sample_name = "0" * (5-len(list(str(sample_counter)))) + str(sample_counter)
        dest_path = os.path.join(target_dir, "images", sample_name + 'png')
        shutil.copy(src_path, dest_path)

        # create query file
        frame_id = int(sample.split(".")[0]) - 1
        imu_curr = imu_data[frame_id] 
        ship_pose = [imu_curr[3],imu_curr[4],imu_curr[2]]
        buoys_on_tile = buoyGTData.getBuoyLocations(ship_pose[0], ship_pose[1]) 
        filteredGT = filterBuoys(ship_pose, buoys_on_tile)
        queries = createQueryData(ship_pose, filteredGT)
        queryFile = os.path.join(target_dir, "queries", sample_name + 'txt')
        with open(queryFile, 'w') as f:
            data = [str(i) + " " + str(data[0]) + " " + str(data[1]) + " " + str(data[2]) + " " + str(data[3]) + "\n" for i,data in enumerate(queries)]
            f.writelines(data)

        # create labels file
        src_path = os.path.join(labels, sample+".json")
        label_data = json.load(open(src_path, 'r'))
        txtlabels = labelsJSON2Yolo(label_data, queries)
        labelfile = os.path.join(target_dir, "labels", sample_name + '.txt')
        with open(labelfile, 'w') as f:
            f.writelines(txtlabels)

        sample_counter += 1

print("DONE!")
