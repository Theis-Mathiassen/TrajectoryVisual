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
