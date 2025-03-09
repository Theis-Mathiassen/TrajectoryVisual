import os
import sys

os.path
currentPath = os.getcwd()
newPath = currentPath + "\src"
sys.path.append(newPath)

from load import load_Tdrive, build_Rtree
from src.knnQuery import KnnQuery
from src.Util import ParamUtil

load_Tdrive("trimmed_small_train.csv")
Rtree_, Trajectories = build_Rtree("trimmed_small_train.csv", "test")

print(Rtree_)

paramUtil = ParamUtil(Rtree_, Trajectories)

params = paramUtil.knnParams(Rtree_)

print(params)

myQuery = KnnQuery(params)

results = myQuery.run(Rtree_)

for t in results:
    print(len(t.nodes))

myQuery.distribute(Trajectories, results)

updatedTrajcs = [t for t in Trajectories if t.id in [x.id for x in results]]

for t in updatedTrajcs:
    print("\n --------------- \n")
    print("Stats for trajectory id: ", t.id)
    print("Number of nodes: ", len(t.nodes))
    for n in t.nodes:
        print(n.score)