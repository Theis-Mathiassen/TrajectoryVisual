from typing import List
from src import Trajectory, Node, Query, rangeQuery, similarityQuery
import numpy as np

def euc_dist_diff(bbox1, bbox2) : 
    # Calculate center of bounding box for the query
    centerx = (bbox2[0] + bbox2[3])/2
    centery = (bbox2[1] + bbox2[4])/2
    centert = (bbox2[2] + bbox2[5])/2

    # Distance measures all 3 dimensions, but maybe the time dimension will simply dominate since that number is so much larger. 
    return np.sqrt(np.power(bbox1[0]-centerx, 2) + np.power(bbox1[1]-centery, 2) + np.power(bbox1[2]-centert, 2)) 

def distribute_Points (trajectories: List[Trajectory], matches, query: Query) : 
    '''Function takes list of trajectories that are stored and distributes points to nodes in each respective trajectory that also appears in "matches".
    matches is an object containing trajectory id, node id, and bounding box for all nodes intersecting the query. 
    The function will distribute one point to a node in a trajectory based on which is closest to the center of the bounding box for the query.'''
    
    
    if type(query) is rangeQuery :
        min_x = query.x1
        min_y = query.y1
        min_t = query.t1
        max_x = query.x2
        max_y = query.y2
        max_t = query.t2
        q_bbox = [min_x, min_y, min_t, max_x, max_y, max_t]
    
    if type(query) is similarityQuery :
        # This part should somehow yield a bounding box for the whole trajectory, but how to do that? 
        NotImplementedError()
    
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

    for key, value in point_dict.items() :
        print(f"Distributing 1 point for trajectory: {key} with node: {value[0]}")
        # TODO: Add functionality here 
