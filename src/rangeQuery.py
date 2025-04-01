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

    
    def distribute(self, trajectories, hits):
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

    def shared_equally(self, trajectories, matches, share_one) : 
        '''Share an amount of points with all m nodes in the trajectory that appeared in the range query.
        With share_one = True, all nodes share 1 point. If share_one = False, all nodes receive one point'''
        def give_point(trajectory: Trajectory, node_ids, amount) :
            for n in trajectory.nodes :
                for m in node_ids : 
                    if n.id == m :
                        n.score += amount/(len(node_ids))
        # Key = Trajectory id, value = (Node id list)
        point_dict = dict()

        # Get matches into correct format
        matches = [(n.object, n.bbox) for n in matches]

        # Put all nodes belonging to each trajectory together in a dict
        for obj, bbox in matches : 
            # If we see trajectory id for the first time, make it a key pointing to empty list in dict before adding node id to said list
            if obj[0] not in point_dict :
                point_dict[obj[0]] = []
            point_dict[obj[0]].append(obj[1])
        
        # Loop through trajectories and check if it appears in the dictionary
        for t in trajectories : 
            #print("T id: ", t.id)
            if t.id in point_dict : 
                if share_one == True :
                    amount = 1
                else : 
                    amount = len(point_dict[t.id])
                give_point(t, point_dict[t.id], amount)


    def gradient_points(self, trajectories, matches) :
        def calculate_point(bbox) : 
            # TODO: Find another way to implement calculate_point. This shit aint workin
            # Points for x and y direction gradient (currently linear and x,y independent). Found by taking 1 subtracted by the normalized distance to the center
            x_dir_point = 1 - (2*np.abs(bbox[0]-self.centerx)/(np.abs(self.x1-self.x2))) 
            y_dir_point = 1 - (2*np.abs(bbox[1]-self.centery)/(np.abs(self.y1-self.y2)))

            # Weighted gradient (currently x = y = 1/2)
            return 1/2 * x_dir_point + 1/2 * y_dir_point 
        
        def give_point(trajectory : Trajectory, node_id, amount) :
            for n in trajectory.nodes:
                if n.id == node_id :
                    n.score += amount

        # Key = Trajectory id, value = (Node id, distance)
        point_dict = dict()

        # Get matches into correct format
        matches = [(n.object, n.bbox) for n in matches]

        for obj, bbox in matches : 
            if obj[0] not in point_dict : 
                point_dict[obj[0]] = []
            point_dict[obj[0]].append((obj[1], bbox)) # Add value (Node id, bbox) to list at index key Trajectory_id

        for t in trajectories :
            if t.id in point_dict :
                for n_id, bbox in point_dict[t.id] : 
                    give_point(t, n_id, calculate_point(bbox))
