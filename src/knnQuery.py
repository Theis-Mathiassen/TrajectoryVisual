from src.Query import Query
from src.Node import Node
from src.Trajectory import Trajectory
from src.Util import DTWDistance, DTWDistanceWithScoring
import math
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
        self.trajectories = params["trajectories"]


    def run(self, rtree):
        # Finds trajectory segments that match the time window of the query
        return self.run2(rtree)
        originSegment = self.getSegmentInTimeWindow(self.trajectory)
        listOfTrajectorySegments = []

        hits = list(rtree.intersection((self.x1, self.y1, self.t1, self.x2, self.y2, self.t2), objects="raw"))
        # Reconstruct trajectories
        trajectories = {}
        # For each node
        for hit in hits:
            # Extract node info
            trajectory_id, node_id = hit
            x = self.trajectories.get(trajectory_id).nodes.data[node_id].x
            y = self.trajectories.get(trajectory_id).nodes.data[node_id].y
            t = self.trajectories.get(trajectory_id ).nodes.data[node_id].t


            # Ignore origin trajectory
            if trajectory_id == self.trajectory.id:
                continue

            node = Node(node_id, x, y, t)

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

    def run2(self, rtree):

        import time

        previous = time.time()
        print("\n --- Running KNN Query ---")

        hits = list(rtree.intersection((self.x1, self.y1, self.t1, self.x2, self.y2, self.t2), objects="raw"))
        
        hits = [(trajectory_id, node_id) for (trajectory_id, node_id) in hits if trajectory_id != self.trajectory.id]
        
        trajectories = {}
        
        for trajectory_id, node_id in hits:
            if trajectory_id not in trajectories:
                trajectories[trajectory_id] = []
            trajectories[trajectory_id].append(node_id)
        
        
        for trajectory in trajectories:
            #boundingNodes = [min(trajectories[trajectory], max(trajectories[trajectory]))]
            minIndex = min(trajectories[trajectory])
            maxIndex = max(trajectories[trajectory])
            trajectories[trajectory] = self.trajectories[trajectory].nodes[minIndex : maxIndex + 1]
            
        
        if len(trajectories.keys()) <= self.k:
            return [Trajectory(id, nodes) for id, nodes in trajectories.items()]
        
        listOfTrajectorySegments = trajectories.items()

        now = time.time()
        print("Time to get trajectories: ", now - previous)
        previous = now

        # Use DTW distance to compute similarity
        similarityMeasures = {}

        
        # Must be of type trajectory to be accepted
        originSegmentTrajectory = self.trajectory

        for segment in listOfTrajectorySegments:
            segmentTrajectory = Trajectory(segment[0], segment[1])
            similarityMeasures[segment[0]] = DTWDistance(originSegmentTrajectory, segmentTrajectory)

        now = time.time()
        print("Time to compute similarity no heuristic: ", now - previous)
        previous = now

        # Sort by most similar, where the most similar have the smallest value
        similarityMeasures = sorted(similarityMeasures.items(), key=lambda x: x[1], reverse=False)


        # get top k ids
        topKIdsReal = [x[0] for x in similarityMeasures[:self.k]]

        
        # Use DTW distance to compute similarity
        similarityMeasures = {}


        # Estimate similarity with heuristic of runtime of T * N rather than T * N^2 as in DTW
        sortedTrajectorySegments = []
        originLength = len(originSegmentTrajectory.nodes)

        for trajectory in listOfTrajectorySegments:
            trajectoryLength = len(trajectory[1])
            diff = 0
            for i in range(min(originLength, trajectoryLength)):
                diff += abs(originSegmentTrajectory.nodes.compressed()[i].x - trajectory[1].compressed()[i].x)
                diff += abs(originSegmentTrajectory.nodes.compressed()[i].y - trajectory[1].compressed()[i].y)
                diff += abs(originSegmentTrajectory.nodes.compressed()[i].t - trajectory[1].compressed()[i].t)
            
            # If the trajectories are of different lengths, add the difference in length to the similarity
            differenceInLength = abs(originLength - trajectoryLength)
            average = diff / min(originLength, trajectoryLength)
            diff += (differenceInLength * average)

            sortedTrajectorySegments.append((trajectory[0], trajectory[1], diff))

        listOfTrajectorySegments = sorted(sortedTrajectorySegments, key=lambda x: x[2])
        
        # listOfTrajectorySegments = [(x[0], x[1]) for x in sortedTrajectorySegments]
         
        # Only look through the first k * 30 of the trajectories
        length = len(listOfTrajectorySegments)
        cutOff = min(self.k * 30, length - 1)

        print("Length: ", length)

        # If more are within similar distance, include them
        valueForCut = listOfTrajectorySegments[cutOff][2] * 1.5
        while listOfTrajectorySegments[cutOff][2] < valueForCut:
            cutOff += 1
            if cutOff >= length:
                break 

        print("Cut off: ", cutOff)

        # Convert
        listOfTrajectorySegments = [(x[0], x[1]) for x in listOfTrajectorySegments[:cutOff]]

        now = time.time()
        print("Time to get heuristic: ", now - previous)
        previous = now


        for segment in listOfTrajectorySegments:
            segmentTrajectory = Trajectory(segment[0], segment[1])
            similarityMeasures[segment[0]] = DTWDistance(originSegmentTrajectory, segmentTrajectory)

        now = time.time()
        print("Time to compute similarity: ", now - previous)
        previous = now

        # Sort by most similar, where the most similar have the smallest value
        similarityMeasures = sorted(similarityMeasures.items(), key=lambda x: x[1], reverse=False)


        # get top k ids
        topKIds = [x[0] for x in similarityMeasures[:self.k]]

        trajectories_output = [Trajectory(trajectory_id, nodes) for trajectory_id, nodes in trajectories.items()]

        now = time.time()
        print("Top k ids before: ", topKIdsReal, "\nTop k ids after:  ", topKIds)
        print("Time to sort: ", now - previous, "\n")



        # Get top k trajectories
        return [x for x in trajectories_output if x.id in topKIds]
        

    def distribute(self, trajectories, matches):

        # Current implementation supports DTW

        originSegment = self.getSegmentInTimeWindow(self.trajectory)

        originSegmentTrajectory = Trajectory(-1, originSegment)

        for trajectory in matches:

            ## Match trajectory to supplied list
            #trajectoryIndex = 0
            #for index, trajectoryInList in enumerate(trajectories.values()):
            #    if trajectoryInList.id == trajectory.id:
            #        trajectoryIndex = index
            #        break

            # Get segment
            #segment = self.getSegmentInTimeWindow(trajectory)
            #segmentTrajectory = Trajectory(trajectory.id, segment)

            # get scorings for nodes (With DTW) in dictionary form
            nodeScores = DTWDistanceWithScoring(originSegmentTrajectory, trajectory)


            # Find relevant nodes and add scores. Note that scores are sorted by index in segment
            for nodeIndex, score in nodeScores.items():
                for index, node in enumerate(trajectories[trajectory.id].nodes):
                    if node.id == trajectory.nodes[nodeIndex].id:    # Found relevant node
                        trajectories[trajectory.id].nodes[index].score += score
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
            #print("Only one node in trajectory with id: " + str(trajectory.id))
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
    