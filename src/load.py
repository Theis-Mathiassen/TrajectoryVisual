from rtree import index
import numpy as np
import pandas as pd
import os

import shutil
import copy

from ast import literal_eval
import json
from src.Util import lonLatToMetric
from tqdm import tqdm

from Node import Node
from Trajectory import Trajectory

CHUNKSIZE = 10**5
PAGESIZE = 16000

#Function to load the Taxi dataset, convert columns and trim it. 
#TO DO: 
#The whole function should be refactored such that functions are applied in chunks. Right now reading the csv gives swap-hell..
def load_Tdrive(src : str, filename="") : 
    tqdm.pandas()

    cwd = os.getcwd()

    #Refactor to use a chunksize instead!
    df = pd.read_csv(os.path.join(cwd, 'datasets', src), delimiter=',')

    #Preprocessing 
    df = df.drop(columns=['CALL_TYPE','ORIGIN_CALL','ORIGIN_STAND','TAXI_ID', 'DAY_TYPE'])

    for index in range(len(df)) : 
        if df['MISSING_DATA'][index] == "True" :
            df.drop(index=index)
    df = df.drop(columns=['MISSING_DATA'])
    
    print("Eval polyline...")
    df["POLYLINE"] = df["POLYLINE"].progress_apply(json.loads)
    print("Done!")
    print("Convert lon lat to metric...")
    df["POLYLINE"] = df["POLYLINE"].progress_apply(rowLonLatToMetric)
    print("Done!")
    # map lonlat as preprocessing
    #df['POLYLINE'] = df['POLYLINE'].apply(lambda a: list(map(lonLatToMetric, a)) )
    
    #Save trimmed data 
    if filename == '' : 
        if os.path.exists(os.path.join(cwd, 'datasets', 'default.csv')) : 
            os.remove(os.path.join(cwd, 'datasets', 'default.csv'))
        df.to_csv(path_or_buf=os.path.join(cwd, 'datasets', 'default.csv'))
    else :
        if os.path.exists(os.path.join(cwd, 'datasets', filename)) :
            os.remove(os.path.join(cwd, 'datasets', filename))
        df.to_csv(path_or_buf=os.path.join(cwd, 'datasets', filename))
    

def load_Tdrive_Rtree(filename=""):
    dataset = "TDrive.csv"
    # Read csv file as dataframe
    tqdm.pandas()
    cwd = os.getcwd()
    path = os.path.join(cwd, 'datasets', dataset)
    df = pd.read_csv(path, converters={'POLYLINE' : json.loads})
    
    # Set up properties
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    p.leaf_capacity = 1000
    p.pagesize = PAGESIZE
    #p.filename = filename

    if filename=='' :
        print("No filename!")
        Rtree_ = index.Index(properties=p)
    else :
        if os.path.exists(filename+'.dat'):
            os.remove(filename+'.dat')
            print('remove', filename+'.dat')
        if os.path.exists(filename+'.idx'):
            os.remove(filename+'.idx')
            print('remove', filename+'.idx')
    
    """ print("Eval polyline...")
    df["POLYLINE"] = df["POLYLINE"].progress_apply(json.loads)
    print("Done!") """
    
    polylines = np.array(df['POLYLINE'])
    trip_ids = np.array(df['TRIP_ID'])
    
    print("Creating rtree..")
    Rtree_ = index.Index(filename, TDriveDataStream(polylines, trip_ids), properties=p)
    print("Done!")
    
    print("Creating trajectories..")
    c = 0
    delete_rec = {}
    Trajectories = []
    length = len(trip_ids)
    for i in tqdm(range(length)):
        nodes = []
        for x, y, t in polylines[i]:
            nodes.append(Node(c, x, y, t))
            c += 1
        Trajectories.append(Trajectory(trip_ids[i], nodes))        
    
        
    return Rtree_, Trajectories



#Build a rtree from csv file and the corresponding trajectories. It is assumed that the csv file contains columns 'TIMESTAMP', 'TRIP_ID' and 'POLYLINE'.
#Function is in two parts: Creating the rtree and creating trajectories.
#Creating the rtree: Happens via bulk load through generator function (datastream). This is a lot faster than inserting each point.
#Creating trajectories: Create a unique trajectory with id = 'TRIP_ID' and nodes = 'POLYLINE' -> Nodes objects
#TO DO:
#Include pagesize param (property of p) for optimizing rtree accesses.
#Reading the csv in chunks if possible would maybe improve performance, unless this hinders bulk loading the rtree.
def build_Rtree(dataset, filename='') :
    # Read csv file as dataframe
    tqdm.pandas()
    cwd = os.getcwd()
    path = os.path.join(cwd, 'datasets', dataset)
    df = pd.read_csv(path, converters={'POLYLINE' : json.loads})
    
    # Set up properties
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    p.leaf_capacity = 1000
    p.pagesize = PAGESIZE
    #p.filename = filename

    if filename=='' :
        print("No filename!")
        Rtree_ = index.Index(properties=p)
    else :
        if os.path.exists(filename+'.dat'):
            os.remove(filename+'.dat')
            print('remove', filename+'.dat')
        if os.path.exists(filename+'.idx'):
            os.remove(filename+'.idx')
            print('remove', filename+'.idx')
    
    """ print("Eval polyline...")
    df["POLYLINE"] = df["POLYLINE"].progress_apply(json.loads)
    print("Done!") """
    
    polylines = np.array(df['POLYLINE'])
    timestamps = np.array(df['TIMESTAMP'])
    trip_ids = np.array(df['TRIP_ID'])
    
    print("Creating rtree..")
    Rtree_ = index.Index(filename, datastream(polylines, timestamps, trip_ids), properties=p)
    print("Done!")
    
    print("Creating trajectories..")
    c = 0
    delete_rec = {}
    Trajectories = []
    length = len(trip_ids)
    for i in tqdm(range(length)):
        t = 0
        nodes = []
        for x, y in polylines[i]:
            nodes.append(Node(c, x, y, timestamps[i] + t*15))
            c += 1
            t += 1
        Trajectories.append(Trajectory(trip_ids[i], nodes))        
    
        
    return Rtree_, Trajectories

#Convert a polyline row in lon lat coordinates to metric coordinates
def rowLonLatToMetric(row):
    nodes = []
    for x, y in row:
        nodes.append(list(lonLatToMetric(x, y)))
    return nodes

#Function to load existing rtree and create a copy of it.
#TO DO:
#Fix it. Dont think it works, but maybe it isn't neccesary?
def loadRtree(originalRtree : index.Index, rtreeName : str, trajectories):
    print("Loading Rtree..")
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    #p.filename = rtreeName
    #p.overwrite = True
    #bounds = originalRtree.bounds
    #points = list(originalRtree.intersection((bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]), objects=True))
    rtreeCopy = index.Index(rtreeName, pointStream(originalRtree), properties=p)
    print("Done!")
    #rtreeCopy.insert(pointStream(originalRtree))
    trajectoriesCopy = copy.deepcopy(trajectories)
    return rtreeCopy, trajectoriesCopy

    
#Generator function taking a dataframe with columns 'TIMESTAMP', 'TRIP_ID' and 'POLYLINE'.
#Yields a rtree point for each point in each polyline
def datastream(polylines, timestamps, trip_ids):
    c = 0
    length = len(trip_ids)
    for i in tqdm(range(length)) :
        t = 0
        timestamp = timestamps[i]
        for x, y in polylines[i] :
            obj=(trip_ids[i], c)
            curTimestamp = timestamp + (15*t)
            yield (c, (x, y, curTimestamp, x, y, curTimestamp), obj)
            
            c+=1
            t+=1

def TDriveDataStream(polylines, trip_ids) :
    c = 0
    length = len(trip_ids)
    for i in tqdm(range(length)) :
        for x, y, t in polylines[i] :
            obj=(trip_ids[i], c)
            yield (c, (x, y, t, x, y, t), obj)
            c+=1

#Generator function taking a rtree
#Yields all points of the rtree 
def pointStream(rtree : index.Index):
    bounds = rtree.bounds
    points = rtree.intersection(coordinates=(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]), objects=True)
    for i, point in enumerate(tqdm(points)):
        yield (i, tuple(point.bbox), i)


def tDriveToCsv():
    """
    Function to convert the T-Drive dataset into a single csv file with TRIP_ID, POLYLINE columns.
    """
    cwd = os.getcwd()

    directory = os.fsencode(os.path.join(cwd, 'datasets','taxi_log_2008_by_id'))
        
    csvdf = pd.DataFrame(columns=['TRIP_ID', 'POLYLINE'])

    taxiIdx = 0

    super_x = []

    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        
        df = pd.read_csv(os.path.join(os.fsdecode(directory),filename), header=None, delimiter=',', names=['id', 'time', 'lon', 'lat'], parse_dates=['time'], date_format="%Y%m%d%H%M%S") #, date_format="%Y%m%d%H%M%S"
        
        if len(df) == 0: 
            continue
        df['time'] = pd.to_datetime(df['time'], yearfirst=True).astype(int) // 10**9
        polyline = df.apply(lambda row : (row['lon'], row['lat'], row['time']), axis=1)
        super_x.append([df['id'][0], polyline.tolist()])
        #csvdf = pd.concat([pd.DataFrame({'TRIP_ID' : df['id'][0], 'POLYLINE' : polyline}, columns=['TRIP_ID', 'POLYLINE']), csvdf], ignore_index=True)
        #csvdf.loc[taxiIdx] = [df['id'][0], polyline]
        taxiIdx += 1
        
    csvdf = pd.concat([csvdf, pd.DataFrame(super_x, columns=['TRIP_ID', 'POLYLINE'])], ignore_index=True, axis=0)
    csvdf.to_csv(path_or_buf=os.path.join(cwd, 'TDrive'), sep=',', index=False)

def datastreamTriple(polylines, trip_ids):
    """
    datastream where polylines is an array of (x, y, t) coordinates.

    Args:
        polylines ([(x : float, y : float, t : float)..]): Array of coordinates  
        trip_ids (int): Trip id / trajectory id.

    Yields:
        Index.item(): tuple representing the rtree item with object
    """
    c = 0
    length = len(trip_ids)
    for i in tqdm(range(length)) :
        for x, y, t in polylines[i] :
            obj=(trip_ids[i], c)
            yield (c, (x, y, t, x, y, t), obj)
            c+=1




if __name__ == "__main__":
    tDriveToCsv()
    """ load_Tdrive("trimmed_small_train.csv")
    Rtree_, Trajectories = build_Rtree("trimmed_small_train.csv", "test")

    hits = list(Rtree_.intersection((-8.66,41.13, 1372636858-2, -8.618643,41.17, 1372637303+100), objects=True))
    print("(Trajectory ID, Node id) pair for intersecting trajectories on range query : ")
    print([(n.object) for n in hits]) """

