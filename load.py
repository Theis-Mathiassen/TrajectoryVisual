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

from src.Node import Node
from src.Trajectory import Trajectory

def load_Tdrive(filename="") : 

    #data = np.genfromtxt("datasets/train.csv", delimiter=',')

    df = pd.read_csv("datasets/train.csv", delimiter=',')

    #Preprocessing 
    df = df.drop(columns=['CALL_TYPE','ORIGIN_CALL','ORIGIN_STAND','TAXI_ID', 'DAY_TYPE'])

    for index in range(len(df)) : 
        if df['MISSING_DATA'][index] == "True" :
            df.drop(index=index)
    df = df.drop(columns=['MISSING_DATA'])
    
    df["POLYLINE"] = df["POLYLINE"].apply(json.loads)
    # map lonlat as preprocessing
    
    #Save trimmed data 
    cwd = os.getcwd()
    if filename == '' : 
        if os.path.exists(os.path.join(cwd, 'datasets', 'default.csv')) : 
            os.remove(os.path.join(cwd, 'datasets', 'default.csv'))
        df.to_csv(path_or_buf=os.path.join(cwd, 'datasets', 'default.csv'))
    else :
        if os.path.exists(os.path.join(cwd, 'datasets', filename)) :
            os.remove(os.path.join(cwd, 'datasets', filename))
        df.to_csv(path_or_buf=os.path.join(cwd, 'datasets', filename))
    

def build_Rtree(dataset, filename='') :
    # Read csv file as dataframe
    cwd = os.getcwd()
    path = os.path.join(cwd, 'datasets', dataset)
    df = pd.read_csv(path)
    #df["POLYLINE"] = df["POLYLINE"].apply(json.loads)
    
    # Set up properties
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
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
    

    c = 0
    delete_rec = {}
    Trajectories = []
    
    print("Creating rtree..")
    Rtree_ = index.Index(filename, datastream(df), properties=p)
    print("Done!")
    
    print("Creating trajectories..")
    for i in tqdm(range(len(df))):
        t = 0
        nodes = []
        for x, y in df['POLYLINE'][i]:
            # x, y = lonLatToMetric(x, y)
            nodes.append(Node(c, x, y, t*15))
            
            c += 1
            t += 1
        Trajectories.append(Trajectory(df['TRIP_ID'][i], nodes))        
    
        
    """ for i in tqdm(range(len(df))) :
        t = 0
        nodes = []
        for x,y in df["POLYLINE"][i] :
            x,y = lonLatToMetric(x,y) # Convert to meters

            Rtree_.insert(c, (x, y, df["TIMESTAMP"][i]+(15*t), x, y, df["TIMESTAMP"][i]+(15*t)), obj=(df["TRIP_ID"][i], c))
            nodes.append(Node(c, x, y, t*15))

            c+=1
            t+=1
        
        Trajectories.append(Trajectory(df["TRIP_ID"][i], nodes))
 """
    return Rtree_, Trajectories


# TEST
#load_Tdrive("trimmed_small_train.csv")
#Rtree_ = build_Rtree("trimmed_small_train.csv", "test")
#
#hits = list(Rtree_.intersection((-8.66,41.13, 1372636858-2, -8.618643,41.17, 1372637303+100), objects=True))
#print("(Trajectory ID, Node id) pair for intersecting trajectories on range query : ")
#print([(n.object, n.bbox) for n in hits])
#

def loadRtree(originalRtree : index.Index, rtreeName : str, trajectories):#srcFilename : str, dstFilename, trajectories):
    print("Loading Rtree..")
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    #p.filename = rtreeName
    #p.overwrite = True
    #bounds = originalRtree.bounds
    #points = list(originalRtree.intersection((bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]), objects=True))
    rtreeCopy = index.Index(pointStream(originalRtree), properties=p)
    print("Done!")
    #rtreeCopy.insert(pointStream(originalRtree))
    trajectoriesCopy = copy.deepcopy(trajectories)
    return rtreeCopy, trajectoriesCopy

    

def datastream(df):
    c = 0
    for i in tqdm(range(len(df))) :
        t = 0
        timestamp = df["TIMESTAMP"][i]
        for x, y in df["POLYLINE"][i] :
            #x, y = lonLatToMetric(x, y) # Convert to meters
            obj=(df["TRIP_ID"][i], c)
            yield (c, (x, y, timestamp + (15*t), x, y, timestamp + (15*t)), obj)
            
            c+=1
            t+=1

def pointStream(rtree : index.Index):
    bounds = rtree.bounds
    points = rtree.intersection(coordinates=(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]), objects=True)
    for i, point in enumerate(points):
        yield (i, tuple(point.bbox), i)


def copyRtreeDatabase(srcName, dstName):
    try: 
        shutil.copy(src=srcName+".dat", dst=dstName+".dat")
        shutil.copy(src=srcName+".idx", dst=dstName+".idx")
        print("Succesfully copied the file")
    except:
        print("Something went wrong when copying the file!")




if __name__ == "__main__":
    load_Tdrive("trimmed_small_train.csv")
    Rtree_, Trajectories = build_Rtree("trimmed_small_train.csv", "test")

    hits = list(Rtree_.intersection((-8.66,41.13, 1372636858-2, -8.618643,41.17, 1372637303+100), objects=True))
    print("(Trajectory ID, Node id) pair for intersecting trajectories on range query : ")
    print([(n.object) for n in hits])

