from src.Trajectory import Trajectory
from src.Query import Query
from src.Node import Node, NodeDiff
import numpy as np
import copy

import random
from rtree import index
import math
import numpy as np

# Util to init params for different query types.
class ParamUtil:
    # init the params to be the bounding box of the Rtree and fix some delta
    def __init__(self, rtree: index.Index, trajectories: dict, delta = 0, k = 3, eps = 1, linesMin = 3):
        boundingBox = rtree.bounds
        
        # Establish mbr for the whole Rtree
        self.xMin = boundingBox[0]
        self.xMax = boundingBox[3]
        self.yMin = boundingBox[1]
        self.yMax = boundingBox[4]
        self.tMin = boundingBox[2]
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
        randomTrajectory: Trajectory = random.choice(list(self.trajectories.values()))
        centerNode: Node = randomTrajectory.nodes[len(randomTrajectory.nodes) // 2] # May be deleted depending on choice of range query generation
        centerX = centerNode.x
        centerY = centerNode.y
        tMin = max(centerNode.t - temporalWindowSize, self.tMin)
        tMax = min(centerNode.t + temporalWindowSize, self.tMax)
        xMin = max(centerX - centerToEdge, self.xMin)
        xMax = min(centerX + centerToEdge, self.xMax)
        yMin = max(centerY - centerToEdge, self.yMin)
        yMax = min(centerY + centerToEdge, self.yMax)
        """ tMin = self.tMin
        tMax = self.tMax
        xMin = self.xMin
        xMax = self.xMax
        yMin = self.yMin
        yMax = self.yMax """
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = self.delta, k = self.k, origin = randomTrajectory, eps = self.eps, linesMin = self.linesMin, trajectories = self.trajectories)
    
    def similarityParams(self, rtree: index.Index, delta = 5000, temporalWindowSize = 5400):
        randomTrajectory: Trajectory = random.choice(list(self.trajectories.values()))
        tMin = randomTrajectory.nodes[0].t
        tMax = randomTrajectory.nodes[-1].t
        xMin = self.xMin
        xMax = self.xMax
        yMin = self.yMin
        yMax = self.yMax
        delta = delta
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = delta, k = self.k, origin = randomTrajectory, eps = self.eps, linesMin = self.linesMin, trajectories = self.trajectories)
    
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
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = self.delta, k = self.k, origin = self.origin, eps = eps, linesMin = linesMin, trajectories = self.trajectories)
    
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
            return np.sqrt(np.power(p1['x']-p2['x'], 2) + np.power(p1['y']-p2['y'], 2))
            return np.sqrt(np.power(p1.x-p2.x, 2) + np.power(p1.y-p2.y, 2))
            return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[1]-p2[1], 2)) 
def euc_dist_diff_3d(p1, p2) : 
            return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[1]-p2[1], 2) + np.power(p1[2]-p2[2], 2)) 

def lcss(epsilon, delta, origin : Trajectory, trajectory : Trajectory) :
    '''Function takes an epsilon (spatial distance) and delta (temporal distance), and two trajectories for comparison. This function uses
    euclidean distance measure for determining the relation to the epsilon.'''

    # Make deep copies
    origin = copy.deepcopy(origin)
    trajectory = copy.deepcopy(trajectory)

    # Then call the actual function. Do this to only deep copy once

    lcssActual(epsilon, delta, origin, trajectory)

def lcssActual(epsilon, delta, origin : Trajectory, trajectory : Trajectory) :

    
    # Prepare variables, but only if they are needed later.
    if (len(origin.nodes) != 0 and len(trajectory.nodes) != 0) :
        ogHead = [origin.nodes[0].x, origin.nodes[0].y]
        tHead = [trajectory.nodes[0].x, trajectory.nodes[0].y]
        timeDiff = np.abs(origin.nodes[0].t - trajectory.nodes[0].t)
        
    if (len(origin.nodes) == 0 or len(trajectory.nodes) == 0) :
        return 0
    elif (euc_dist_diff_2d(ogHead, tHead) <= epsilon and (timeDiff <= delta)) : 
        newOrigin = origin 
        newOrigin.nodes.pop(0)
        newTrajectory = trajectory
        newTrajectory.nodes.pop(0)
        return lcss(epsilon, delta, newOrigin, newTrajectory)+1
    else :
        newOrigin = origin
        newOrigin.nodes.pop(0)
        newTrajectory = trajectory
        newTrajectory.nodes.pop(0)
        return max(lcss(epsilon, delta, newOrigin, trajectory), lcss(epsilon, delta, newTrajectory, origin))


# Idea is to find the optimal route for insert delete matches. 
# Then the nodes in this path are rewarded based on how little they contributed to overall cost
def DTWDistanceWithScoring(origin : Trajectory, other : Trajectory) -> int:
    originNodes = origin.nodes
    otherNodes = other.nodes
    DTW = np.ndarray((len(originNodes),len(otherNodes)))
    pathTracker = np.ndarray((len(originNodes),len(otherNodes)), dtype=int) #Keeps track of min path (Insert, delete, match)
    costTracker = np.ndarray((len(originNodes),len(otherNodes))) #Keeps track of cost. This could recalculated later, optimizing for time atm
    
    w = abs(len(originNodes) - len(otherNodes)) + 1
    
    for i in range(len(originNodes)):
        for j in range(len(otherNodes)):
            DTW[i, j] = math.inf
            
    DTW[0, 0] = 0
    
    for i in range(1, len(originNodes)):
        for j in range(max(1, i-w), min(len(otherNodes), i+w)):
            DTW[i, j] = 0
            pathTracker[i, j] = 0
            
    for i in range(1, len(originNodes)):
        for j in range(max(1, i-w), min(len(otherNodes), i+w)):
            cost = euc_dist_diff_2d(originNodes[i], otherNodes[j])

            minimum = min(  DTW[i-1  , j     ],  # insertion
                            DTW[i    , j-1   ],  # deletion
                            DTW[i-1  , j-1   ])  # match

            DTW[i, j] = cost + minimum

            costTracker[i, j] = cost

            # Accounts for edge case of the existance of several min paths
            if minimum == DTW[i - 1, j]: 
                pathTracker[i, j] += 1
            if minimum == DTW[i, j - 1]:
                pathTracker[i, j] += 2
            if minimum == DTW[i - 1, j - 1]:
                pathTracker[i, j] += 4

    #Retrace steps, and find each (x,y) along the optimal route visited
    visited = get_visited(pathTracker, len(originNodes), len(otherNodes))

    # Find total cost and each node cost contribution
    totalCost = 0
    nodeCosts = {}
    for (x,y) in visited:
        cost = costTracker[x, y]
        totalCost += cost

        if y not in nodeCosts:
            nodeCosts[y] = cost
        else:
            nodeCosts[y] += cost


    nodeScores = {}
    # We give each node a point, but minus by their cost contribution
    for (x,y) in visited:
        cost = nodeCosts[y]

        costContribution = cost / totalCost

        score = pow(1 - costContribution, 2)

        if y not in nodeScores:
            nodeScores[y] = score
        else:
            nodeScores[y] += score

    return nodeScores
    
def get_visited(pathTracker, length_x, length_y):
    toVisit = [(length_x - 1, length_y - 1)]
    visited = []

    # Go through list and add onto till empty
    while (len(toVisit) != 0):
        (x, y) = toVisit.pop(-1)


        # Continue if already visited or reached edge
        if (x, y) in visited or x == 0 or y == 0:
            continue

        visited.append((x, y))

        path = int(pathTracker[x, y])

        if path & 1: # If insert
            toVisit.append((x - 1, y))
        if path & 2: # If Deletion
            toVisit.append((x, y - 1))
        if path & 4: # If match
            toVisit.append((x - 1, y - 1))
    
    return visited