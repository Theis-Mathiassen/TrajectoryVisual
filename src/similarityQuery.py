from rtree import index
import os
import numpy as np
from Trajectory import Trajectory
from Node import Node
from Query import Query


# Temp data

import random # Remove later


def build_Random_Rtree(filename='', points_per_trajectory = 10, trajectory_amount = 5) :
    p = index.Property()
    p.dimension = 3
    #p.dat_extension = 'data'
    #p.idx_extension = 'index'
    if filename=='' :
        Rtree_ = index.Index(properties=p)
    else :
        if os.path.exists(filename+'.dat'):
            os.remove(filename+'.dat')
            print('remove', filename+'.dat')
        if os.path.exists(filename+'.idx'):
            os.remove(filename+'.idx')
            print('remove', filename+'.idx')
        Rtree_ = index.Index(filename, properties=p)

    id = 0

    for c in range (trajectory_amount):
        x = 0
        y = 0
        for i in range(points_per_trajectory):
            t = 5 * i
            x += random.uniform(0,5)
            y += random.uniform(0,5)

            Rtree_.insert(c, (x, y, t, x, y, t), obj=(id, c))
            id += 1

    return Rtree_

Rtree_ = build_Random_Rtree(filename="test")

hits = list(Rtree_.intersection((0, 0, 0, 10, 3, 5), objects=True))
print([(n.object) for n in hits])
print([(n.bbox) for n in hits])


# Actual algo


class SimilarityQuery(Query):
    trajectory: Trajectory
    t1: float
    t2: float
    delta: float
    
    def __init__(self, params):
        self.trajectory = params["trajectory"]
        self.t1 = params["t1"]
        self.t2 = params["t2"]
        self.delta = params["delta"]

    def run(self, rtree):

        # For each node in the trajectory
        for node in self.trajectory.nodes:

            x = node.x 
            y = node.y 
            t = node.t

            trajectory_hits = {}
            # If fits time range
            if (self.t1 <= t and t <= self.t2):
                point1 = np.array((node.x, node.y))

                # Get all points possibly within range with range query
                maybe_hits = list((rtree.intersection((x + self.delta, y + self.delta, t, x - self.delta, y - self.delta, t), objects=True)))

                for maybe_hit in maybe_hits:
                    (x_idx, y_idx, t_idx, _, _, _) = maybe_hit.bbox

                    if t != t_idx:
                        break
                    
                    point2 = np.array((x_idx, y_idx))

                    # calculating Euclidean distance to find if actual hit
                    dist = np.linalg.norm(point1 - point2)
                    
                    if dist > self.delta:
                        break

                    (node_id, trajectory_id) = maybe_hit.object
                    node = Node(node_id, x_idx, y_idx, t_idx)

                    # Get list of nodes by trajectories
                    if trajectory_id not in trajectory_hits:
                        trajectory_hits[trajectory_id] = []

                    trajectory_hits[trajectory_id].append(node)

        # Convert to corrct format
        trajectories = [Trajectory(trajectory_id, nodes) for trajectory_id, nodes in trajectory_hits.items()]
        
        return trajectories


bounds 
