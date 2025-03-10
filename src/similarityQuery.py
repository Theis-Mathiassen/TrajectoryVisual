from rtree import index
import os
import numpy as np
from Trajectory import Trajectory
from Node import Node
from Query import Query

class SimilarityQuery(Query):
    trajectory: Trajectory
    t1: float
    t2: float
    delta: float
    
    def __init__(self, params):
        self.trajectory = params["origin"]
        self.t1 = params["t1"]
        self.t2 = params["t2"]
        self.delta = params["delta"]

    def run(self, rtree):

        # For each node in the trajectory
        trajectory_hits = {}
        for node in self.trajectory.nodes:
            x = node.x 
            y = node.y 
            t = node.t

            # If fits time range
            if (self.t1 <= t and t <= self.t2):
                point1 = np.array((node.x, node.y))

                # Get all points possibly within range with range query
                maybe_hits = list((rtree.intersection((x - self.delta, y - self.delta, t, x + self.delta, y + self.delta, t), objects=True)))

                for maybe_hit in maybe_hits:
                    (x_idx, y_idx, t_idx, _, _, _) = maybe_hit.bbox
                    (node_id, trajectory_id) = maybe_hit.object

                    # If fit time, and not part of the queried trajectory
                    if t == t_idx and trajectory_id != self.trajectory.id:
                        point2 = np.array((x_idx, y_idx))

                        # calculating Euclidean distance to find if actual hit
                        dist = np.linalg.norm(point1 - point2)
                        if dist <= self.delta:
                            node = Node(node_id, x_idx, y_idx, t_idx)

                            # Get list of nodes by trajectories
                            if trajectory_id not in trajectory_hits:
                                trajectory_hits[trajectory_id] = []

                            trajectory_hits[trajectory_id].append(node)

        # Convert to corrct format
        trajectories = [Trajectory(trajectory_id, nodes) for trajectory_id, nodes in trajectory_hits.items()]
        
        return trajectories
    
    def distribute(self, trajectories, matches):
        return super().distribute(trajectories, matches)