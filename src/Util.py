from Trajectory import Trajectory
from Query import Query
from Node import Node, NodeDiff
import numpy as np


import random
from rtree import index
import math
import numpy as np

# Util to init params for different query types.
class ParamUtil:
    # init the params to be the bounding box of the Rtree and fix some delta
    def __init__(self, rtree: index.Index, trajectories: list[Trajectory], delta = 0, k = 3, eps = 1, linesMin = 3):
        boundingBox = rtree.bounds()
        
        # Establish mbr for the whole Rtree
        self.xMin = boundingBox[0]
        self.xMax = boundingBox[1]
        self.yMin = boundingBox[2]
        self.yMax = boundingBox[3]
        self.tMin = boundingBox[4]
        self.tMax = boundingBox[5]
        
        # List of trajectories and their nodes. Note that every element of trajectory.nodes may not be in a simplified Rtree
        self.trajectories = trajectories
        
        # Delta for similarity queries
        self.delta = delta
        # k for knn queries
        self.k = k
        # eps and minimum number of lines in a cluster for cluster queries
        self.eps = eps
        self.linesMin = linesMin
        
        self.origin: Trajectory = None
        
    
    
    # The following presents different functions to generate params (dictionary) for the different types of queries. 
    # Note that some values are None and needs changing depending on how we choose queries
    def rangeParams(self, rtree: index.Index, centerToEdge = 1000, temporalWindowSize = 5400):
        randomTrajectory: Trajectory = random.choice(self.trajectories)
        centerNode: Node = randomTrajectory.nodes[len(randomTrajectory.nodes) // 2] # May be deleted depending on choice of range query generation
        centerX = centerNode.x
        centerY = centerNode.y
        tMin = centerNode.t - temporalWindowSize
        tMax = centerNode.t + temporalWindowSize
        xMin = centerX - centerToEdge
        xMax = centerX + centerToEdge
        yMin = centerY - centerToEdge
        yMax = centerY + centerToEdge
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = self.delta, k = self.k, origin = randomTrajectory, eps = self.eps, linesMin = self.linesMin)
    
    def similarityParams(self, rtree: index.Index, delta = 5000):
        randomTrajectory: Trajectory = random.choice(self.trajectories)
        tMin = randomTrajectory.nodes[0].t 
        tMax = randomTrajectory.nodes[-1].t
        xMin = self.xMin
        xMax = self.xMax
        yMin = self.yMin
        yMax = self.yMax
        delta = delta
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = delta, k = self.k, origin = randomTrajectory, eps = self.eps, linesMin = self.linesMin)
    
    def knnParams(self, rtree: index.Index, k = 3):
        randomTrajectory: Trajectory = random.choice(self.trajectories)
        tMin = self.tMin 
        tMax = self.tMax
        xMin = self.xMin
        xMax = self.xMax
        yMin = self.yMin
        yMax = self.yMax
        k = k
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = self.delta, k = k, origin = randomTrajectory, eps = self.eps, linesMin = self.linesMin)
    
    def clusterParams(self, rtree: index.Index):
        tMin = self.tMin 
        tMax = self.tMax
        xMin = self.xMin
        xMax = self.xMax
        yMin = self.yMin
        yMax = self.yMax
        k = self.k
        eps = None
        linesMin = None
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = self.delta, k = self.k, origin = self.origin, eps = eps, linesMin = linesMin)
    
def lonLatToMetric(lon, lat):   #top answer https://stackoverflow.com/questions/1253499/simple-calculations-for-working-with-lat-lon-and-km-distance
    north = lat * 110574.0
    east = lon * 111320.0 * math.cos(0.017453292519943295*lat)
    return east, north

# Based on wikipedia article on dynamic time warping https://en.wikipedia.org/wiki/Dynamic_time_warping
# but changed to a dynamic window size such that we always can compare two trajectories
def DTWDistance(origin : Trajectory, other : Trajectory) -> int:
    originNodes = origin.nodes
    otherNodes = other.nodes
    DTW = np.ndarray((len(originNodes),len(otherNodes)))
    
    w = abs(len(originNodes) - len(otherNodes)) + 1
    
    for i in range(len(originNodes)):
        for j in range(len(otherNodes)):
            DTW[i, j] = math.inf
            
    DTW[0, 0] = 0
    
    for i in range(1, len(originNodes)):
        for j in range(max(1, i-w), min(len(otherNodes), i+w)):
            DTW[i, j] = 0
            
    for i in range(1, len(originNodes)):
        for j in range(max(1, i-w), min(len(otherNodes), i+w)):
            cost = euc_dist_diff_2d(originNodes[i], otherNodes[j])
            DTW[i, j] = cost + min(DTW[i-1  , j     ],  # insertion
                                   DTW[i    , j-1   ],  # deletion
                                   DTW[i-1  , j-1   ])  # match
    
    return DTW[len(originNodes)-1, len(otherNodes)-1]

def euc_dist_diff_2d(p1, p2) : 
            # Distance measures all 3 dimensions, but maybe the time dimension will simply dominate since that number is so much larger. 
            return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[1]-p2[1], 2)) 
def euc_dist_diff_3d(p1, p2) : 
            # Distance measures all 3 dimensions, but maybe the time dimension will simply dominate since that number is so much larger. 
            return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[1]-p2[1], 2) + np.power(p1[2]-p2[2], 2)) 

