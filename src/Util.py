from src.Trajectory import Trajectory
from src.Query import Query
from src.Node import Node, NodeDiff
import numpy as np
import copy

import random
from rtree import index
import math
from numba import vectorize, float64,float32, jit, njit, types, prange

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
    def rangeParams(self, rtree: index.Index, centerToEdge = 1000, temporalWindowSize = 5400, flag = 2, index = None, cell = None):
        if cell is not None:
            centerX = cell[0] * 2 * centerToEdge + centerToEdge
            centerY = cell[1] * 2 * centerToEdge + centerToEdge
            centerT = cell[2] * 2 * temporalWindowSize + temporalWindowSize
            randomTrajectory = None
        else:
            if not index:
                randomTrajectory: Trajectory = random.choice(list(self.trajectories.values()))
            else:
                randomTrajectory: Trajectory = self.trajectories[index] #self.trajectories.keys()[index]
            centerNode: Node = randomTrajectory.nodes[len(randomTrajectory.nodes) // 2] # May be deleted depending on choice of range query generation
            centerX = centerNode.x
            centerY = centerNode.y
            centerT = centerNode.t
        tMin = max(centerT - temporalWindowSize, self.tMin)
        tMax = min(centerT + temporalWindowSize, self.tMax)
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
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = self.delta, k = self.k, origin = randomTrajectory, eps = self.eps, linesMin = self.linesMin, trajectories = self.trajectories, flag = flag)
    
    def gaussianRangeParams(self, point, centerToEdge = 1000, temporalWindowSize = 5400, flag = 2):
        centerX = point[0]
        centerY = point[1]
        centerT = point[2]
        tMin = max(centerT - temporalWindowSize, self.tMin)
        tMax = min(centerT + temporalWindowSize, self.tMax)
        xMin = max(centerX - centerToEdge, self.xMin)
        xMax = min(centerX + centerToEdge, self.xMax)
        yMin = max(centerY - centerToEdge, self.yMin)
        yMax = min(centerY + centerToEdge, self.yMax)
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = self.delta, k = self.k, origin = None, eps = self.eps, linesMin = self.linesMin, trajectories = self.trajectories, flag = flag)


    def similarityParams(self, rtree: index.Index, delta = 5000, temporalWindowSize = 5400, index = None, nodeIndex = None):
        if not index:
            randomTrajectory: Trajectory = random.choice(list(self.trajectories.values()))
        else:
            randomTrajectory: Trajectory = self.trajectories[index]
        
        if not nodeIndex :
            centerNode: Node = randomTrajectory.nodes[len(randomTrajectory.nodes) // 2]
        else:
            centerNode: Node = randomTrajectory.nodes[nodeIndex]
        centerTime = centerNode.t
        minId = 0
        maxId = len(randomTrajectory.nodes) - 1
        for node in randomTrajectory.nodes[0:len(randomTrajectory.nodes) // 2]:
            if node.t >= centerTime - temporalWindowSize // 2:
                minId = node.id
                break
            
        for node in randomTrajectory.nodes[len(randomTrajectory.nodes) // 2 : -1]:
            if node.t >= centerTime + temporalWindowSize // 2:
                maxId = node.id - 1
                break
            
        
        tMin = max(self.tMin, centerTime - temporalWindowSize // 2)
        tMax = min(self.tMax, centerTime + temporalWindowSize // 2)
        
        xMax = max(map(lambda node : node.x, randomTrajectory.nodes.data[minId:maxId + 1]))
        xMax = min(xMax + delta, self.xMax)
        xMin = min(map(lambda node : node.x, randomTrajectory.nodes.data[minId:maxId + 1]))
        xMin = max(xMin - delta, self.xMin)
        
        yMax = max(map(lambda node : node.y, randomTrajectory.nodes.data[minId:maxId + 1]))
        yMax = min(yMax + delta, self.yMax)
        yMin = min(map(lambda node : node.y, randomTrajectory.nodes.data[minId:maxId + 1]))
        yMin = max(yMin - delta, self.yMin)
        
        delta = delta
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = delta, k = self.k, origin = randomTrajectory, eps = self.eps, linesMin = self.linesMin, trajectories = self.trajectories)
    
    def knnParams(self, rtree: index.Index, k = 3, temporalWindowSize = 5400, spatialWindowSize = 1000, index = None, nodeIndex = None, cell = None):
        if not index:
            randomTrajectory: Trajectory = random.choice(list(self.trajectories.values()))
        else:
            randomTrajectory: Trajectory = self.trajectories[index]
        
        if not nodeIndex :
            centerNode: Node = randomTrajectory.nodes[len(randomTrajectory.nodes) // 2]
        else:
            centerNode: Node = randomTrajectory.nodes[nodeIndex]
        centerTime = centerNode.t
        
        #tMax = self.tMax
        if cell is not None:
            centerX = cell[0] * 2 * spatialWindowSize + spatialWindowSize
            centerY = cell[1] * 2 * spatialWindowSize + spatialWindowSize
            centerT = cell[2] * 2 * temporalWindowSize + temporalWindowSize
            xMin = centerX - spatialWindowSize
            xMax = centerX + spatialWindowSize
            yMin = centerY - spatialWindowSize
            yMax = centerY + spatialWindowSize
            tMin = centerT - temporalWindowSize
            tMax = centerT + temporalWindowSize
        else:
            xMin = self.xMin
            xMax = self.xMax
            yMin = self.yMin
            yMax = self.yMax
            tMin = max(self.tMin, centerTime - temporalWindowSize // 2)
            tMax = min(self.tMax, centerTime + temporalWindowSize // 2)
            
        
        k = k
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = self.delta, k = k, origin = randomTrajectory, eps = self.eps, linesMin = self.linesMin, trajectories = self.trajectories)
    
    def clusterParams(self, rtree: index.Index, temporalWindowSize = 5400, minLines = 2, spatialWindowSize = 1000, index = None, eps = None, cell = None):
        if not index:
            randomTrajectory: Trajectory = random.choice(list(self.trajectories.values()))
        else:
            randomTrajectory: Trajectory = self.trajectories[index]
        if cell:
            centerX = cell[0] * 2 * spatialWindowSize + spatialWindowSize
            centerY = cell[1] * 2 * spatialWindowSize + spatialWindowSize
            centerT = cell[2] * 2 * temporalWindowSize + temporalWindowSize
            xMin = centerX - spatialWindowSize
            xMax = centerX + spatialWindowSize
            yMin = centerY - spatialWindowSize
            yMax = centerY + spatialWindowSize
            tMin = centerT - temporalWindowSize
            tMax = centerT + temporalWindowSize
        centerNode: Node = randomTrajectory.nodes[len(randomTrajectory.nodes) // 2] # May be deleted depending on choice of range query generation
        centerX = centerNode.x
        centerY = centerNode.y
        tMin = max(centerNode.t - temporalWindowSize, self.tMin)
        tMax = min(centerNode.t + temporalWindowSize, self.tMax)
        xMin = max(centerX - spatialWindowSize, self.xMin)
        xMax = min(centerX + spatialWindowSize, self.xMax)
        yMin = max(centerY - spatialWindowSize, self.yMin)
        yMax = min(centerY + spatialWindowSize, self.yMax)
        eps = eps
        linesMin = minLines
        return dict(t1 = tMin, t2= tMax, x1 = xMin, x2 = xMax, y1 = yMin, y2 = yMax, delta = self.delta, k = self.k, origin = randomTrajectory, eps = eps, linesMin = linesMin, trajectories = self.trajectories, centerToEdge = spatialWindowSize, temporalWindowSize = temporalWindowSize)
    
def lonLatToMetric(lon, lat):   #top answer https://stackoverflow.com/questions/1253499/simple-calculations-for-working-with-lat-lon-and-km-distance
    north = lat * 110574.0
    east = lon * 111320.0 * math.cos(0.017453292519943295*lat)
    return east, north

def metricToLonLat(east, north):
    lat = north / 110574.0
    lon = east / (111320.0*math.cos(0.017453292519943295*lat))
    return lon, lat

# Based on wikipedia article on dynamic time warping https://en.wikipedia.org/wiki/Dynamic_time_warping
# but changed to a dynamic window size such that we always can compare two trajectories
def DTWDistance(origin : Trajectory, other : Trajectory) -> int:
    originNodes = origin.nodes
    otherNodes = other.nodes.compressed()
    if len(originNodes) == 0 or len(otherNodes) == 0:
        return math.inf
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
            if originNodes is list :
                ogbox = dict({'x' : originNodes[i].x, 'y' : originNodes[i].y, 't' : originNodes[i].t})
            else : 
                ogbox = dict({'x' : originNodes.data[i].x, 'y' : originNodes.data[i].y, 't' : originNodes.data[i].t})
            otherbox = dict({'x' : otherNodes[j].x, 'y' : otherNodes[j].y, 't' : otherNodes[j].t})

            cost = euc_dist_diff_2d(ogbox , otherbox )
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

    return lcssActual(epsilon, delta, origin, trajectory)

def lcssActual(epsilon, delta, origin : Trajectory, trajectory : Trajectory) :

    
    # Prepare variables, but only if they are needed later.
    if (len(origin.nodes) != 0 and len(trajectory.nodes) != 0) :
        ogHead = dict({'x': origin.nodes.data[0].x, 'y' : origin.nodes.data[0].y})
        tHead = dict({'x' : trajectory.nodes.data[0].x, 'y': trajectory.nodes.data[0].y})
        timeDiff = np.abs(origin.nodes.data[0].t - trajectory.nodes.data[0].t)
        
    if (len(origin.nodes) == 0 or len(trajectory.nodes) == 0) :
        return 0
    elif (euc_dist_diff_2d(ogHead, tHead) <= epsilon and (timeDiff <= delta)) : 
        newOrigin = origin 
        newOrigin.nodes = newOrigin.nodes[1:]
        newTrajectory = trajectory
        newTrajectory.nodes = newTrajectory.nodes[1:]
        return lcssActual(epsilon, delta, newOrigin, newTrajectory)+1
    else :
        newOrigin = origin
        newOrigin.nodes = newOrigin.nodes[1:]
        newTrajectory = trajectory
        newTrajectory.nodes = newTrajectory.nodes[1:]
        return max(lcssActual(epsilon, delta, newOrigin, trajectory), lcssActual(epsilon, delta, newTrajectory, origin))


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
            cost = euc_dist_diff_2d(dict({'x' : originNodes[i].x, 'y' : originNodes[i].y, 't' : originNodes[i].t}), dict({'x' : otherNodes[j].x, 'y' : otherNodes[j].y, 't' : otherNodes[j].t}))

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

@jit(float64(float64[:], float64[:,:]), nopython=True, cache=True)
def spatial_distance(node, nodes):

    dx = np.abs(nodes[:,0] - node[0])
    dy = np.abs(nodes[:,1] - node[1])

    distances = np.sqrt(dx**2 + dy**2)

    return np.min(distances)

@jit(float64(float64[:], float64[:,:]), nopython=True, cache=True)
def temporal_distance(node, nodes):
    
    distances = np.abs(nodes[:,2] - node[2])

    return np.min(distances)



@jit(float64(float64[:,:], float64[:,:]), cache=True, looplift=True)
def get_distancesSpatial(evalNodes, referenceNodes):
        spatial_similarity = 0
        evalLength = len(evalNodes)

        for idx in range(evalLength):
            spatial_similarity += spatial_distance(evalNodes[idx], referenceNodes)



        spatial_similarity = spatial_similarity / evalLength

        return spatial_similarity


@jit(float64(float64[:,:], float64[:,:]), cache=True, looplift=True)
def get_distancesTemporal(evalNodes, referenceNodes):
        temporal_similarity = 0
        evalLength = len(evalNodes)

        for idx in range(evalLength):
            temporal_similarity += temporal_distance(evalNodes[idx], referenceNodes)


        temporal_similarity = temporal_similarity / evalLength

        return temporal_similarity


def spatio_temporal_linear_combine_distance(originTrajectory : Trajectory, otherTrajectory : Trajectory, weight):
    if len(originTrajectory.nodes.compressed()) == 0 or len(otherTrajectory.nodes) == 0:
        return float('inf')

    originNodes = originTrajectory.nodes.compressed()
    otherNodes = otherTrajectory.nodes

    otherNodes = [x for x in otherNodes if x is not np.ma.masked]

    npOrigin = np.array([[n.x, n.y, n.t] for n in originNodes])
    npOther = np.array([[n.x, n.y, n.t] for n in otherNodes])
    
    return spatio_temporal_linear_combine_distance_real(npOrigin, npOther, weight)
    
# Based on https://www.vldb.org/pvldb/vol10/p1178-shang.pdf
@jit(float64(float64[:,:], float64[:,:], float64), nopython=True, cache=True, fastmath=True)
def spatio_temporal_linear_combine_distance_real(npOrigin, npOther, weight):
    """
    Gets the spatio-temporal distance between two lists of nodes

    weight is a value between 0 and 1 determining how much the spatio-temporal distance is weighted

    result = spatial dist * weight + temporal dist * (1 - weight)`
    """
    #originNodes = originTrajectory.nodes.compressed()
    #otherNodes = otherTrajectory.nodes.compressed()

    # if len(originNodes) == 0 or len(otherNodes) == 0:
    #     return float('inf') # If no nodes, return infinity so least likely to be selected. Also avoids errors

    


    spatial_similarity_1 = get_distancesSpatial(npOrigin, npOther)
    temporal_similarity_1 = get_distancesTemporal(npOrigin, npOther)
    spatial_similarity_2 = get_distancesSpatial(npOther, npOrigin)
    temporal_similarity_2 = get_distancesTemporal(npOther, npOrigin)


    spatial_similarity = spatial_similarity_1 + spatial_similarity_2
    temporal_similarity = temporal_similarity_1 + temporal_similarity_2

    return spatial_similarity * weight + temporal_similarity * (1 - weight)



def spatio_temporal_linear_combine_distance_with_scoring(originTrajectory : Trajectory, otherTrajectory : Trajectory, weight, nodesToAward):
    """
    We give points to each node where it is the minimum distance. Divided by the distance

    We only loop over which nodes are closest to the origin trajectory. 
    Not which from the origin trajectory are closest to others, as we are rewarding others

    We also factor the alpha weight in
    """
    origin_nodes = originTrajectory.nodes.compressed()
    other_nodes = otherTrajectory.nodes

    npOrigin = np.array([[n.x, n.y, n.t] for n in origin_nodes])
    npOther = np.array([[n.x, n.y, n.t] for n in other_nodes])

    for origin_node in npOrigin:
        for func in [spatial_distance_func, temporal_distance_func]:
            closestNodeIndex, dist = func(origin_node, npOther)

            if dist < 1: # Set distance to a minimum of 1
                dist = 1

            nodeID = otherTrajectory.nodes[closestNodeIndex].id

            length = len(nodesToAward)
            if nodeID >= length:
                print("Node ID is bigger than length of nodes to award")
                print("Node ID: " + str(nodeID) + " Length: " + str(length))

            node = nodesToAward[nodeID]

            node.score['knn'] += weight / dist
            #otherTrajectory.nodes[closestNodeIndex].score['knn'] += weight / dist

           
def get_min_dist_node(origin_node, nodes, func):
    min = math.inf
    closestIndex = None
    for index, node in enumerate(nodes[0:]):
        dist = func(origin_node, node)
        if dist < min:
            min = dist
            closestIndex = index

    return closestIndex, min

# Like the ones above but also return the index of the min val
@jit(nopython=True, cache=True)
def temporal_distance_func(node, other_nodes):
    
    distances = np.abs(other_nodes[:,2] - node[2])

    min_idx = np.argmin(distances)
    return min_idx, np.min(distances)


@jit(nopython=True, cache=True)
def spatial_distance_func(node, other_nodes):

    dx = np.abs(other_nodes[:,0] - node[0])
    dy = np.abs(other_nodes[:,1] - node[1])

    distances = np.sqrt(dx**2 + dy**2)

    min_idx = np.argmin(distances)
    return min_idx, np.min(distances)


def getGaussianDist(avgVals, stdDeviation = 500):
    """
    We expect avgVals to be an array of 3

    stdDeviation defaults to 500
    """
    result = np.random.normal(avgVals, stdDeviation, size=(1, 3))
    return result

