from rtree import index
from Trajectory import Trajectory
from Node import Node
from Query import Query
import numpy as np
import rangeQuery

class RangeQuery(Query):
    x1: float
    y1: float
    x2: float
    y2: float
    t1: float
    t2: float
    flag: int
    
    def __init__(self, params):
        self.x1 = params["x1"]
        self.y1 = params["y1"]
        self.t1 = params["t1"]
        self.x2 = params["x2"]
        self.y2 = params["y2"]
        self.t2 = params["t2"]
        self.flag = params["flag"]

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
        matches is an object containing trajectory id, node id, and bounding box for all nodes intersecting the query.'''
        match self.flag :
            case 1 :
                return self.winner_takes_all(self, trajectories, matches)
            case 2 : 
                return self.shared_equally(self, trajectories, matches)



    def winner_takes_all(self, trajectories, matches):
        '''Give 1 point to the node closest to the origin for the range query for each trajectory'''
        def give_point(trajectory: Trajectory, node_id) :
            for n in trajectory.nodes :
                if n.id == node_id :
                    n.score += 1

        # TODO: Get center points from query
        centerx = 0
        centery = 0
        centert = 0
        q_bbox = [centerx, centery, centert]

        # Key = Trajectory id, value = (Node id, distance)
        point_dict = dict()

        # Get matches into correct format
        matches = [(n.object, n.bbox) for n in matches]

        for obj, bbox in matches : 
            dist_current = rangeQuery.euc_dist_diff(bbox, q_bbox)

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

    def shared_equally(self, trajectories, matches) : 
        '''Give 1/m points to all m nodes in the trajectory that appeared in the range query'''
        def give_point(trajectory: Trajectory, node_ids) :
            for n in trajectory.nodes :
                for m in node_ids : 
                    if n.id == m :
                        n.score += 1/(len(node_ids))
        # Key = Trajectory id, value = (Node id list)
        point_dict = dict()

        # Get matches into correct format
        matches = [(n.object, n.bbox) for n in matches]

        # Put all nodes belonging to each trajectory together in a dict
        for obj in matches : 
            # If we see trajectory id for the first time, make it a key pointing to empty list in dict before adding node id to said list
            if obj[0] not in point_dict :
                point_dict[obj[0]] = []
            point_dict[obj[0]].append(obj[1])

        # Loop through trajectories and check if it appears in the dictionary
        for t in trajectories : 
            if t in point_dict : 
                give_point(t, point_dict[t])

