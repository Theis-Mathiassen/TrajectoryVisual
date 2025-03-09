from Query import Query
from Node import Node
from Trajectory import Trajectory
import math
from Util import DTWDistance
import numpy as np

class KnnQuery(Query):
    trajectory: Trajectory
    t1: float
    t2: float
    
    def __init__(self, params):
        self.trajectory = params["origin"]
        self.k = params["k"]
        self.t1 = params["t1"]
        self.t2 = params["t2"]
        self.x1 = params["x1"]
        self.x2 = params["x2"]
        self.y1 = params["y1"]
        self.y2 = params["y2"]

    def run(self, rtree):
        # Finds trajectory segments that match the time window of the query

        originSegment = self.getSegmentInTimeWindow(self.trajectory)
        listOfTrajectorySegments = []

        hits = list(rtree.intersection((self.x1, self.y1, self.t1, self.x2, self.y2, self.t2), objects=True))
        # Reconstruct trajectories
        trajectories = {}
        # For each node
        for hit in hits:
            # Extract node info
            x_idx, y_idx, t_idx, _, _, _ = hit.bbox
            trajectory_id, node_id = hit.object

            # Ignore origin trajectory
            if trajectory_id == self.trajectory.id:
                continue

            node = Node(node_id, x_idx, y_idx, t_idx)

            # Get list of nodes by trajectories
            if trajectory_id not in trajectories:
                trajectories[trajectory_id] = []

            trajectories[trajectory_id].append(node)

        # If fewer than k 
        if len(trajectories.keys()) <= self.k:
            return [Trajectory(id, nodes) for id, nodes in trajectories.items()]

        listOfTrajectorySegments = trajectories.items()

        # Use DTW distance to compute similarity
        similarityMeasures = {}
        
        # Must be of type trajectory to be accepted
        originSegmentTrajectory = Trajectory(-1, originSegment)
        for segment in listOfTrajectorySegments:
            segmentTrajectory = Trajectory(segment[0], segment[1])
            similarityMeasures[segment[0]] = DTWDistance(originSegmentTrajectory, segmentTrajectory)

        # Sort by most similar, where the most similar have the smallest value
        similarityMeasures = sorted(similarityMeasures.items(), key=lambda x: x[1], reverse=False)

        # get top k ids
        topKIds = [x[0] for x in similarityMeasures[:self.k]]

        trajectories_output = [Trajectory(trajectory_id, nodes) for trajectory_id, nodes in trajectories.items()]



        # Get top k trajectories
        return [x for x in trajectories_output if x.id in topKIds]

    def distribute(self, trajectories, matches):

        # Current implementation supports DTW

        originSegment = self.getSegmentInTimeWindow(self.trajectory)

        originSegmentTrajectory = Trajectory(-1, originSegment)

        for trajectory in matches:

            # Match trajectory to supplied list
            trajectoryIndex = 0
            for index, trajectoryInList in enumerate(trajectories):
                if trajectoryInList.id == trajectory.id:
                    trajectoryIndex = index
                    break

            # Get segment
            segment = self.getSegmentInTimeWindow(trajectory)
            segmentTrajectory = Trajectory(trajectory.id, segment)

            # get scorings for nodes (With DTW) in dictionary form
            nodeScores = self.DTWDistanceWithScoring(originSegmentTrajectory, segmentTrajectory)


            # Find relevant nodes and add scores. Note that scores are sorted by index in segment
            for nodeIndex, score in nodeScores.items():
                for index, node in enumerate(trajectories[trajectoryIndex].nodes):
                    if node.id == segment[nodeIndex].id:    # Found relevant node
                        trajectories[trajectoryIndex].nodes[index].score += score
                        break

                        

    def getSegmentInTimeWindow(self, trajectory):
        
        length = len(trajectory.nodes)
        time_start = trajectory.nodes[0].t
        time_end = trajectory.nodes[-1].t

        # Check if not in time window
        if (time_start > self.t2 or time_end < self.t1):
            return []
        
        time_per_segment = (time_end - time_start) / length

        if time_per_segment == 0: # If only one node we must do this, also so we do not divide by 0
            print("Only one node in trajectory with id: " + str(trajectory.id))
            return trajectory.nodes
        
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
    
    def get_visited(self, pathTracker, length_x, length_y):
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

    def euc_dist_diff_2d(self, p1, p2) : 
            # Distance measures all 3 dimensions, but maybe the time dimension will simply dominate since that number is so much larger. 
            return np.sqrt(np.power(p1.x-p2.x, 2) + np.power(p1.y-p2.y, 2))
            return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[1]-p2[1], 2)) 

