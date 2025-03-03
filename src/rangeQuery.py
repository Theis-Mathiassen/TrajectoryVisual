from rtree import index
from Trajectory import Trajectory
from Node import Node
from Query import Query
import numpy as np

class RangeQuery(Query):
    x1: float
    y1: float
    x2: float
    y2: float
    t1: float
    t2: float
    
    def __init__(self, params):
        self.x1 = params["x1"]
        self.y1 = params["y1"]
        self.t1 = params["t1"]
        self.x2 = params["x2"]
        self.y2 = params["y2"]
        self.t2 = params["t2"]

    def run(self, rtree):
        # Gets nodes in range query
        hits = list(rtree.intersection((self.x1, self.y1, self.t1, self.x2, self.y2, self.t2), objects=True))

        trajectories = {}
        # For each node
        for hit in hits:
            # Extract node info
            x_idx, y_idx, t_idx, _, _, _ = hit.bbox
            node_id, trajectory_id = hit.object

            node = Node(node_id, x_idx, y_idx, t_idx)
            print(node)

            # Get list of nodes by trajectories
            if trajectory_id not in trajectories:
                trajectories[trajectory_id] = []

            trajectories[trajectory_id].append(node)
        
        trajectories_output = [Trajectory(trajectory_id, nodes) for trajectory_id, nodes in trajectories.items()]

        return trajectories_output
    
    def distribute(self, trajectories, matches):
        '''Function takes list of trajectories that are stored and distributes points to nodes in each respective trajectory that also appears in "matches".
        matches is an object containing trajectory id, node id, and bounding box for all nodes intersecting the query. 
        The function will distribute one point to a node in a trajectory based on which is closest to the center of the bounding box for the query.'''
        def euc_dist_diff(bbox1, bbox2) : 
            # Calculate center of bounding box for the query
            centerx = (bbox2[0] + bbox2[3])/2
            centery = (bbox2[1] + bbox2[4])/2
            centert = (bbox2[2] + bbox2[5])/2

            # Distance measures all 3 dimensions, but maybe the time dimension will simply dominate since that number is so much larger. 
            return np.sqrt(np.power(bbox1[0]-centerx, 2) + np.power(bbox1[1]-centery, 2) + np.power(bbox1[2]-centert, 2)) 
        
        def give_point(trajectory: Trajectory, node_id) :
            for n in trajectory.nodes :
                if n.id == node_id :
                    n.score += 1
        
        q_bbox = [self.x1, self.y1, self.t1, self.x2, self.y2, self.t2]
        # Key = Trajectory id, value = (Node id, distance)
        point_dict = dict()

        # Get matches into correct format
        matches = [(n.object, n.bbox) for n in matches]

        for obj, bbox in matches : 
            dist_current = euc_dist_diff(bbox, q_bbox)

            if obj[0] in point_dict : 
                dist_prev = point_dict.get(obj[0])[1]
                if dist_current <= dist_prev :
                    point_dict[obj[0]] = (obj[1], dist_current)
            else :
                point_dict[obj[0]] = (obj[1], dist_current)

        # TODO: Here we should probably have sorted dictionary and list of trajectories so worst case run time is always N instead of N^2 (not including sort)
        for key, value in point_dict.items() :
            print(f"Distributing 1 point for trajectory: {key} with node: {value[0]}")
            for t in trajectories :
                if t.id == key :
                    give_point(t, value)
