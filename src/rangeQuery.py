from rtree import index
from src.Trajectory import Trajectory
from src.Node import Node
from src.Query import Query
import numpy as np
from src.Util import euc_dist_diff_2d
import rangeQuery


class RangeQuery(Query):
    x1: float
    y1: float
    x2: float
    y2: float
    t1: float
    t2: float
    flag: int
    centerx: float
    centery: float
    centert: float
    
    def __init__(self, params):
        self.x1 = params["x1"]
        self.y1 = params["y1"]
        self.t1 = params["t1"]
        self.x2 = params["x2"]
        self.y2 = params["y2"]
        self.t2 = params["t2"]
        self.trajectories = params["trajectories"]
        self.flag = params["flag"]
        self.centerx = (self.x2+self.x1)/2
        self.centery = (self.y2+self.y1)/2
        self.centert = (self.t2+self.t1)/2


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

    
    def distribute(self, trajectories, matches):
        '''Function takes list of trajectories that are stored and distributes points to nodes in each respective trajectory that also appears in "matches".
        matches is an object containing trajectory id, node id, and bounding box for all nodes intersecting the query.'''
        match self.flag :
            case 1 :
                return self.winner_takes_all(trajectories, matches) # Only 1 node per trajectory in range gets a point
            case 2 : 
                return self.shared_equally(trajectories, matches, share_one=True) # All nodes in range divide 1 point among nodes in trajectory
            case 3 : 
                return self.shared_equally(trajectories, matches, share_one=False) # All nodes in range get 1 point 
            case 4 :
                return self.gradient_points(trajectories, matches) # All nodes in range get points based on proximity to center of range query



    def winner_takes_all(self, trajectories, matches):
        '''Give 1 point to the node closest to the origin for the range query for each trajectory'''
        def give_point(trajectory: Trajectory, node_id) :
            for n in trajectory.nodes :
                if n.id == node_id[0] :
                    n.score += 1


        q_bbox = [self.centerx, self.centery, self.centert]
        q_bbox_dict = {'x' : q_bbox[0], 'y' : q_bbox[1], 't' : q_bbox[2]}
        # Key = Trajectory id, value = (Node id, distance)
        point_dict = dict()

        # Get matches into correct format
        #matches = [(n.object, n.bbox) for n in self.hits]


        for hit in matches : 
            trajectory_id, node_id = hit
            node = self.trajectories.get(trajectory_id).nodes[node_id]

            dist_current = euc_dist_diff_2d(dict({'x' : node.x, 'y' : node.y, 't' : node.t}), q_bbox_dict)

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

    def shared_equally(self, trajectories, matches, share_one) : 
        '''Share an amount of points with all m nodes in the trajectory that appeared in the range query.
        With share_one = True, all nodes share 1 point. If share_one = False, all nodes receive one point'''
        # Key = Trajectory id, value = (Node id list)
        point_dict = dict()

        # Get matches into correct format
        #matches = [(n.object, n.bbox) for n in matches]

        # Put all nodes belonging to each trajectory together in a dict
        for trajectory_id, node_id in matches : 
            # If we see trajectory id for the first time, make it a key pointing to empty list in dict before adding node id to said list
            if trajectory_id not in point_dict :
                point_dict[trajectory_id] = []
            point_dict[trajectory_id].append(node_id)
        
        # Loop through trajectories and check if it appears in the dictionary
        for trajectory_id, nodes_ids in point_dict.items():
            amount = 1
            
            if share_one == True:
                amount /= len(nodes_ids)
                
            for node_id in nodes_ids:
                trajectories[trajectory_id].nodes[node_id].score += amount



    def gradient_points(self, trajectories, matches) :
        # Key = Trajectory id, value = (Node id list)
        point_dict = dict()

        # Get matches into correct format
        #matches = [(n.object, n.bbox) for n in matches]

        # Put all nodes belonging to each trajectory together in a dict
        for trajectory_id, node_id in matches : 
            # If we see trajectory id for the first time, make it a key pointing to empty list in dict before adding node id to said list
            if trajectory_id not in point_dict :
                point_dict[trajectory_id] = []
            point_dict[trajectory_id].append(node_id)

        width = np.abs(self.x1-self.x2)
        height = np.abs(self.y1-self.y2)

        for trajectory_id, nodes_ids in point_dict.items():
            for node_id in nodes_ids:
                node = trajectories[trajectory_id].nodes[node_id]
                x_dir_point = 1 - (2*np.abs(node.x-self.centerx)/(width)) 
                y_dir_point = 1 - (2*np.abs(node.y-self.centery)/(height))
                amount =  x_dir_point / 2 + y_dir_point / 2
                trajectories[trajectory_id].nodes[node_id].score += amount
