from Trajectory import Trajectory
from Query import Query
from Point import Node

import random
from rtree import index

# Util to init params for different query types.
class ParamUtil:
    # init the params to be the bounding box of the Rtree and fix some delta
    def __init__(self, rtree: index.Index, trajectories: list(Trajectory), delta = 0):
        boundingBox = rtree.bounds()
        
        self.xMin = boundingBox[0]
        self.xMax = boundingBox[1]
        self.yMin = boundingBox[2]
        self.yMax = boundingBox[3]
        self.tMin = boundingBox[4]
        self.tMax = boundingBox[5]
        
        self.trajectories = trajectories
        self.delta = delta
        self.origin: Trajectory = None
        
    
    def RangeParams(self, rtree: index.Index):
        randomTrajectory = random.choice(self.trajectories)
        centerNode: Node = randomTrajectory.nodes[len(randomTrajectory.nodes) // 2]
        