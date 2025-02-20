from rtree import index
import numpy as np
import pandas as pd
import os
from ast import literal_eval

#data = np.genfromtxt("datasets/train.csv", delimiter=',')
print("Hello1")
df = pd.read_csv("datasets/train.csv", delimiter=',')
print("Hello2")
df["POLYLINE"] = df["POLYLINE"].apply(literal_eval)

def build_Rtree(filename='') :
    print("Hello3")
    p = index.Property()
    p.dimension = 3
    #p.dat_extension = 'data'
    #p.idx_extension = 'index'
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
    for i in range(len(df)) :
        if (i % 1000 == 0):
            print(i + " out of: " + len(df))
        t = 0
        for x,y in df["POLYLINE"][i] :
            Rtree_.insert(c, (x, y, df["TIMESTAMP"][i]+(15*t), x, y, df["TIMESTAMP"][i]+(15*t)), obj=(df["TRIP_ID"][i], c))
            c+=1
            t+=1
    return Rtree_
Rtree_ = build_Rtree("test")

hits = list(Rtree_.intersection((-8.66,41.13, 1372636858-2, -8.618643,41.17, 1372637303+100), objects=True))
print([(n.object) for n in hits])
