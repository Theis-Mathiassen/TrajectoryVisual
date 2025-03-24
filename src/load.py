from rtree import index
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import shutil
import copy
from ast import literal_eval
import json
from src.Util import lonLatToMetric
from tqdm import tqdm

from src.Node import Node
from src.Trajectory import Trajectory

CHUNKSIZE = 10**5
PAGESIZE = 16000

#Function to load the Taxi dataset, convert columns and trim it. 
#TO DO: 
#The whole function should be refactored such that functions are applied in chunks. Right now reading the csv gives swap-hell..
#Drop rows with polylines of length 0..
def load_Tdrive(src : str, filename="") : 
    tqdm.pandas()

    cwd = os.getcwd()

    #Refactor to use a chunksize instead!
    df = pd.read_csv(os.path.join(cwd, 'datasets', src), delimiter=',')

    #Preprocessing 
    df = df.drop(columns=['CALL_TYPE','ORIGIN_CALL','ORIGIN_STAND','TAXI_ID', 'DAY_TYPE'])
    
    print("Eval polyline...")
    df["POLYLINE"] = df["POLYLINE"].progress_apply(json.loads)
    print("Done!")

    for index in range(len(df)) : 
        if df['MISSING_DATA'][index] == "True" or len(df['POLYLINE'][index]) == 0:
            df.drop(index=index)
    df = df.drop(columns=['MISSING_DATA'])
    
    print("Convert lon lat to metric...")
    df["POLYLINE"] = df["POLYLINE"].progress_apply(rowLonLatToMetric)
    print("Done!")
    # map lonlat as preprocessing
    
    #Save trimmed data 
    if filename == '' : 
        if os.path.exists(os.path.join(cwd, 'datasets', 'default.csv')) : 
            os.remove(os.path.join(cwd, 'datasets', 'default.csv'))
        df.to_csv(path_or_buf=os.path.join(cwd, 'datasets', 'default.csv'))
    else :
        if os.path.exists(os.path.join(cwd, 'datasets', filename)) :
            os.remove(os.path.join(cwd, 'datasets', filename))
        df.to_csv(path_or_buf=os.path.join(cwd, 'datasets', filename))
    
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
    
    polylines = np.array(df['POLYLINE'])
    timestamps = np.array(df['TIMESTAMP'])
    trip_ids = np.array(df['TRIP_ID'])
    
    print("Creating rtree..")
    if os.path.exists(filename + '.index'):
        Rtree_ = index.Index(filename, properties=p)
    else:
        Rtree_ = index.Index(filename, datastream(polylines, timestamps, trip_ids), properties=p)
    print("Done!")
    
    print("Creating trajectories..")
    c = 0
    delete_rec = {}
    Trajectories = {}
    length = len(trip_ids)
    for i in tqdm(range(length)):
        if len(polylines[i]) == 0:
            pass
        else:
            t = 0
            c = 0
            nodes = [] #np.array(nparrayStream(polylines[i], timestamps[i]))
            for x, y in polylines[i]:
                nodes.append(Node(c, x, y, timestamps[i] + t*15))
                c += 1
                t += 1
            Trajectories.update({int(trip_ids[i]) : Trajectory(int(trip_ids[i]), ma.copy(nodes))})        
    
        
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
def loadRtree(rtreeName : str, trajectories):
    print("Loading Rtree..")
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    p.leaf_capacity = 1000
    p.pagesize = PAGESIZE
    #p.filename = rtreeName
    #p.overwrite = True
    #bounds = originalRtree.bounds
    #points = list(originalRtree.intersection((bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]), objects=True))
    rtreeCopy = index.Index(rtreeName, pointStream(trajectories), properties=p)
    print("Done!")
    #rtreeCopy.insert(pointStream(originalRtree))
    trajectoriesCopy = copy.deepcopy(trajectories)
    return rtreeCopy, trajectoriesCopy

    
#Generator function taking a dataframe with columns 'TIMESTAMP', 'TRIP_ID' and 'POLYLINE'.
#Yields a rtree point for each point in each polyline
def datastream(polylines, timestamps, trip_ids):
    length = len(trip_ids)
    c = 0
    for i in tqdm(range(length)) :
        t = 0
        timestamp = timestamps[i]
        if len(polylines[i]) == 0:
            pass
        else:
            for x, y in polylines[i] :
                obj=(int(trip_ids[i]), t)
                curTimestamp = timestamp + (15*t)
                yield (c, (x, y, curTimestamp, x, y, curTimestamp), obj)
                
                c+=1
                t+=1

#Generator function taking a rtree
#Yields all points of the rtree 
def pointStream(trajectories: dict):
    for trajectory in trajectories.values():
        nodes = trajectory.nodes.compressed()
        for i in range(len(nodes)):
            obj=(trajectory.id, nodes[i].id)
            yield (trajectory.id, (nodes[i].x, nodes[i].y, nodes[i].t, nodes[i].x, nodes[i].y, nodes[i].t), obj)
    

def nparrayStream(polyline, timestamp):
    c = 0
    t = 0
    for x, y in polyline:
            yield Node(c, x, y, timestamp + t*15)
            c += 1
            t += 1

if __name__ == "__main__":
    load_Tdrive("trimmed_small_train.csv")
    Rtree_, Trajectories = build_Rtree("trimmed_small_train.csv", "test")

    hits = list(Rtree_.intersection((-8.66,41.13, 1372636858-2, -8.618643,41.17, 1372637303+100), objects=True))
    print("(Trajectory ID, Node id) pair for intersecting trajectories on range query : ")
    print([(n.object) for n in hits])

