from Query import Query
from Trajectory import Trajectory
import math

class RangeQuery(Query):
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
        listOfTrajectories = []

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
            listOfTrajectories.append(trajectory.id, segment)

        # If amount of segments <= k we return these arbitrarily
        if (len(listOfTrajectories) <= self.k):
            return [trajectory[0] for trajectory in listOfTrajectories] #Returns ids 

        # TODO, find the k least dissimilar

        pass


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

