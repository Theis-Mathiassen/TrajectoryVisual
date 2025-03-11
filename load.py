from rtree import index
import numpy as np
import pandas as pd
import os
from ast import literal_eval
from src.Util import lonLatToMetric

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
    df["POLYLINE"] = df["POLYLINE"].apply(literal_eval)
    
    # Set up properties
    p = index.Property()
    p.dimension = 3

    if filename=='' :
        Rtree_ = index.Index(properties=p)
    else :
        if os.path.exists(filename+'.dat'):
            os.remove(filename+'.dat')
            print('remove', filename+'.dat')
        if os.path.exists(filename+'.idx'):
            os.remove(filename+'.idx')
            print('remove', filename+'.idx')
        Rtree_ = index.Index(filename, properties=p)

    c = 0
    delete_rec = {}
    Trajectories = []
    for i in range(len(df)) :
        t = 0
        nodes = []
        for x,y in df["POLYLINE"][i] :
            x,y = lonLatToMetric(x,y) # Convert to meters

            Rtree_.insert(c, (x, y, df["TIMESTAMP"][i]+(15*t), x, y, df["TIMESTAMP"][i]+(15*t)), obj=(df["TRIP_ID"][i], c))
            nodes.append(Node(c, x, y, t*15))

            c+=1
            t+=1
        
        Trajectories.append(Trajectory(df["TRIP_ID"][i], nodes))

    return Rtree_, Trajectories


# TEST
#load_Tdrive("trimmed_small_train.csv")
#Rtree_ = build_Rtree("trimmed_small_train.csv", "test")
#
#hits = list(Rtree_.intersection((-8.66,41.13, 1372636858-2, -8.618643,41.17, 1372637303+100), objects=True))
#print("(Trajectory ID, Node id) pair for intersecting trajectories on range query : ")
#print([(n.object, n.bbox) for n in hits])
#

if __name__ == "__main__":
    load_Tdrive("trimmed_small_train.csv")
    Rtree_, Trajectories = build_Rtree("trimmed_small_train.csv", "test")

    hits = list(Rtree_.intersection((-8.66,41.13, 1372636858-2, -8.618643,41.17, 1372637303+100), objects=True))
    print("(Trajectory ID, Node id) pair for intersecting trajectories on range query : ")
    print([(n.object) for n in hits])
