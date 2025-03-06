from Query import Query
from Trajectory import Trajectory
import math
from Util import DTWDistanc
import numpy as np

class KnnQuery(Query):
    trajectory: Trajectory
    allTrajectories: list[Trajectory]
    t1: float
    t2: float
    
    def __init__(self, params):
        self.allTrajectories = params["trajectories"]
        self.trajectory = params["origin"]
        self.k = params["k"]
        self.t1 = params["t1"]
        self.t2 = params["t2"]

    def run(self, rtree):
        # Finds trajectory segments that match the time window of the query

        originSegment = self.getSegmentInTimeWindow(self.trajectory)
        listOfTrajectorySegments = []

        # Generate a list of trajectory segments within the time window and their ids
        # Form: (id, segment) : (1: [Node, ..., Node]), (2: [Node, ..., Node]), ...)
        for trajectory in self.allTrajectories:
            # Do not look at origin
            if trajectory.id == self.trajectory.id:
                continue

            segment = self.getSegmentInTimeWindow(trajectory)
            # If empty, therefore not in time window
            if (segment == []):
                continue
            listOfTrajectorySegments.append(trajectory.id, segment)

        # If amount of segments <= k we return these arbitrarily
        if (len(listOfTrajectorySegments) <= self.k):
            return [x[0] for x in listOfTrajectorySegments] #Returns ids 

        # Use DTW distance to compute similarity
        similarityMeasures = {}
        for segment in listOfTrajectorySegments:
            similarityMeasures[segment[0]] = DTWDistanc(originSegment, segment[1])

        # Sort by most similar, where the most similar have the smallest value
        similarityMeasures = sorted(similarityMeasures.items(), key=lambda x: x[1], reverse=False)

        # get top k ids
        topKIds = [x[0] for x in similarityMeasures[:self.k]]

        return [trajectory for trajectory in self.allTrajectories if trajectory.id in topKIds]


    def distribute(self, trajectories, matches):

        # Current implementation supports DTW

        originSegment = self.getSegmentInTimeWindow(self.trajectory)

        for trajectory in matches:

            # Match trajectory to supplied list
            trajectoryIndex = 0
            for index, trajectoryInList in trajectories:
                if trajectoryInList.id == trajectory.id:
                    trajectoryIndex = index
                    break

            # Get segment
            segment = self.getSegmentInTimeWindow(trajectory)

            # get scorings for nodes (With DTW) in dictionary form
            nodeScores = self.DTWDistanceWithScoring(originSegment, segment)

            for nodeId, score in nodeScores.items():
                for nodeIndex, node in trajectories:
                    if nodeId == node.id:
                        trajectories[trajectoryIndex].nodes[nodeIndex].score += score
                        break

                        


    def getSegmentInTimeWindow(self, trajectory):
        
        length = len(trajectory)
        time_start = trajectory[0].t
        time_end = trajectory[-1].t

        # Check if not in time window
        if (time_start > self.t2 or time_end < self.t1):
            return []
        
        time_per_segment = (time_end - time_start) / length
        
        start_index = math.ceil((self.t1 - time_start) / time_per_segment) # Round up so within time window
        end_index = math.floor((self.t2 - time_start) / time_per_segment)  # Round down so within time window

        #  If exceeds boundaries
        if(self.t1 < time_start):
            start_index = 0
        if(self.t2 > time_end):
            end_index = length - 1

        segment = trajectory.nodes[start_index:end_index]

        return segment
    
    # A lot of code copied from Util file
    # Idea is to find the optimal route for insert delete matches. 
    # Then the nodes in this path are rewarded based on how little they contributed to overall cost
    def DTWDistanceWithScoring(self, origin : Trajectory, other : Trajectory) -> int:
        originNodes = origin.nodes
        otherNodes = other.nodes
        DTW = np.ndarray((len(originNodes),len(otherNodes)))
        pathTracker = np.ndarray((len(originNodes),len(otherNodes))) #Keeps track of min path (Insert, delete, match)
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
                cost = self.euc_dist_diff_2d(originNodes[i], otherNodes[j])

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
        visited = self.get_visited(pathTracker, len(originNodes), len(otherNodes))

        totalCost = 0
        nodeCost = {}
        for (x,y) in visited:
            cost = costTracker[x, y]
            totalCost += cost

            if y not in nodeCost:
                nodeCost[y] = cost + 1 # We add one such that they cannot get infinite points if cost 0
            else:
                nodeCost[y] += cost

        nodeScores = {}
        # Nodes are rewarded, minus their contribution to the total cost. Note that nodes who appear several times get more points
        for (x,y) in visited:
            score = totalCost - nodeCost[y] / totalCost
            
            if y not in nodeScores:
                nodeScores[y] = score
            else:
                nodeScores[y] += score

        return nodeScores
    
    def get_visited(pathTracker, length_x, length_y):
        toVisit = [(length_x - 1, length_y - 1)]
        visited = []

        # Go through list and add onto till empty
        while len(toVisit != 0):
            (x, y) = toVisit.pop

            # Continue if already visited or reached edge
            if (x, y) in visited or x == 0 or y == 0:
                continue

            visited.append((x, y))

            path = pathTracker(x, y)

            if path & 1: # If insert
                toVisit.append(x - 1, y)
            if path & 2: # If Deletion
                toVisit.append(x, y - 1)
            if path & 4: # If match
                toVisit.append(x - 1, y - 1)
        
        return visited






    def euc_dist_diff_2d(p1, p2) : 
            # Distance measures all 3 dimensions, but maybe the time dimension will simply dominate since that number is so much larger. 
            return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[1]-p2[1], 2)) 

