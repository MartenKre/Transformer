import numpy as np
from math import sin, cos, radians, asin, sqrt
import pyproj

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

def ECEF2LatLng(x, y, z, rad=False):
    # function returns lat & lng coordinates in Deg
    transformer = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'})
    lon1, lat1, alt1 = transformer.transform(x,y,z,radians=rad)
    return lat1, lon1, alt1

def T_ECEF_Ship(x, y, z, heading):
    # returns transformtion matrix ECEF_T_Ship
    # expects arguments xyz to be ship pos in ECEF and ships heading in deg
    ECEF_T_ENU = T_ECEF_ENU(x,y,z)
    ENU_T_Ship = T_ENU_Ship(heading)

    ECEF_T_SHIP = ECEF_T_ENU @ ENU_T_Ship
    return ECEF_T_SHIP

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

def T_W_Ship(pos, heading):
    # used by rendering to transform ship cs to world (start cs)
    # Arguments: pos of ship (x,y,z) in global cs. heading of ship in rad
    # Start CS is at 0,0,0 and heading = 0 -> x=north, y = west, z = up
    # ship moves around in it
    T = np.eye(4)
    R = getRotationMatrix(-1*heading, 'z')  # -1*heading, since we have to rotate start cs in negative direction for pos heading
    T[:3, :3] = R
    T[:3, 3] = pos
    return T

def extract_RPY(R):
    # extracts roll, pitch yaw from 3x3 rotation matrix
    # returns rpy in radians
    R = np.asarray(R)

    yaw=np.arctan2(R[1,0],R[0,0])
    pitch=np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2))
    roll=np.arctan2(R[2,1],R[2,2])

    return roll, pitch, yaw
