from rtree import index
from src.Trajectory import Trajectory
from src.Node import Node
from src.Query import Query
import numpy as np
from src.Util import euc_dist_diff_2d

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
        self.trajectories = params["trajectories"]

    def run(self, rtree):
        # Gets nodes in range query
        hits = list(rtree.intersection((self.x1, self.y1, self.t1, self.x2, self.y2, self.t2), objects="raw"))

        """ trajectories = {}
        # For each node
        for hit in hits:
            # Extract node info
            trajectory_id, node_id = hit
            x = self.trajectories.get(trajectory_id).nodes[node_id].x
            y = self.trajectories.get(trajectory_id).nodes[node_id].y
            t = self.trajectories.get(trajectory_id ).nodes[node_id].t

            node = Node(node_id, x, y, t)

            # Get list of nodes by trajectories
            if trajectory_id not in trajectories:
                trajectories[trajectory_id] = []

            trajectories[trajectory_id].append(node)
        
        trajectories_output = [Trajectory(trajectory_id, nodes) for trajectory_id, nodes in trajectories.items()]
        #print(len(trajectories_output))
        self.hits = hits """
        return hits
    
    def distribute(self, trajectories, hits):
        '''Function takes list of trajectories that are stored and distributes points to nodes in each respective trajectory that also appears in "matches".
        matches is an object containing trajectory id, node id, and bounding box for all nodes intersecting the query. 
        The function will distribute one point to a node in a trajectory based on which is closest to the center of the bounding box for the query.'''
       
        def give_point(trajectory: Trajectory, node_id) :
            for n in trajectory.nodes :
                if n.id == node_id :
                    n.score += 1
        
        # TODO: Get center points from query
        centerx = (self.x1 + self.x2) / 2
        centery = (self.y1 + self.y2) / 2
        centert = (self.t1 + self.t2) / 2
        q_bbox = dict({'x' : centerx, 'y' : centery, 't' : centert})
        
        # Key = Trajectory id, value = (Node id, distance)
        point_dict = dict()

        # Get matches into correct format
        #matches = [(n.object, n.bbox) for n in self.hits]

        for hit in hits : 
            trajectory_id, node_id = hit
            x = self.trajectories.get(trajectory_id).nodes[node_id].x
            y = self.trajectories.get(trajectory_id).nodes[node_id].y
            t = self.trajectories.get(trajectory_id ).nodes[node_id].t
            dist_current = euc_dist_diff_2d(dict({'x' : x, 'y' : y, 't' : t}), q_bbox)

            if trajectory_id in point_dict : 
                dist_prev = point_dict.get(trajectory_id)[1]
                if dist_current <= dist_prev :
                    point_dict[trajectory_id] = (node_id, dist_current)
            else :
                point_dict[trajectory_id] = (node_id, dist_current)

        # TODO: Here we should probably have sorted dictionary and list of trajectories so worst case run time is always N instead of N^2 (not including sort)
        for key, value in point_dict.items() :
            #print(f"Distributing 1 point for trajectory: {key} with node: {value[0]}")
            give_point(trajectories.get(key), value)
            """ for t in trajectories :
                if t.id == key :
                    give_point(t, value) """
