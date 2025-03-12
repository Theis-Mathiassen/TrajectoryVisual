from Query import Query
from Trajectory import Trajectory
import numpy as np
from TRACLUS import traclus
from sklearn.cluster import OPTICS
from Node import Node
from rtree import index

class ClusterQuery(Query):

    def __init__(self, params):
        super().__init__(params)
        self.t1 = params["t1"]  
        self.t2 = params["t2"]  # t1=start time, t2=end time
        self.eps = params["eps"]  # max distance for clustering
        self.min_lines = params["linesMin"]  # min number of lines in a cluster
        self.origin = params["origin"]  # the query trajectory

    def _trajectory_to_numpy(self, trajectory):
        """Convert a Trajectory object to numpy array format required by TRACLUS."""
        return np.array([[node.x, node.y] for node in trajectory.nodes])

    def _filter_trajectories_by_time(self, trajectories, rtree):
        """Filter trajectories based on temporal constraints."""
        filtered = []
        for hit in rtree.intersection((float('-inf'), float('-inf'), self.t1,
                                     float('inf'), float('inf'), self.t2), objects=True):
            _, trajectory_id = hit.object
            # Get unique trajectory IDs that have points in the time window
            if trajectory_id not in [t.id for t in filtered]:
                matching_traj = next((t for t in trajectories if t.id == trajectory_id), None)
                if matching_traj:
                    filtered.append(matching_traj)
        return filtered

    def run(self, rtree):
        # get all trajectories with points in the time window
        trajectories = self._filter_trajectories_by_time(self.params["trajectories"], rtree)
        
        if not trajectories:
            return []

        # convert trajectories to numpy arrays for TRACLUS
        numpy_trajectories = [self._trajectory_to_numpy(t) for t in trajectories]
        origin_numpy = self._trajectory_to_numpy(self.origin)
        numpy_trajectories.append(origin_numpy)  # Add query trajectory

        # run TRACLUS
        _, _, _, clusters, cluster_assignments, _ = traclus(
            numpy_trajectories,
            max_eps=self.eps,
            min_samples=self.min_lines,
            directional=True,
            use_segments=True,
            clustering_algorithm=OPTICS,
            progress_bar=False
        )

        # find which cluster contains the query trajectory
        query_idx = len(numpy_trajectories) - 1  # last added trajectory is the query
        query_cluster = cluster_assignments[query_idx]

        # get trajectories in the same cluster as the query
        similar_trajectories = []
        for idx, cluster_id in enumerate(cluster_assignments):
            if cluster_id == query_cluster and idx != query_idx:
                similar_trajectories.append(trajectories[idx])

        return similar_trajectories

    def distribute(self, trajectories, matches):
        # current impl is very basic, simply adds 1 to the score of each node in the trajectory
        # todo: make scoring more sophisticated w/ scoring based on directional changes, spatial density etc.?
        for trajectory in trajectories:
            for node in trajectory.nodes:
                node.score += 1

if __name__ == "__main__":
    # create sample trajectories for testing
    def create_sample_trajectory(id, points):
        nodes = [Node(i, x, y, t) for i, (x, y, t) in enumerate(points)]
        return Trajectory(id, nodes)

    # create some sample trajectories
    # traj1 and 2 follow similar paths
    traj1 = create_sample_trajectory(1, [
        (0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)
    ])
    traj2 = create_sample_trajectory(2, [
        (0.1, 0.1, 0), (1.1, 1.1, 1), (2.1, 2.1, 2), (3.1, 3.1, 3)
    ])
    # traj3 follows a different path
    traj3 = create_sample_trajectory(3, [
        (0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3)
    ])

    # create r-tree index
    p = index.Property()
    p.dimension = 3  # 3D index (x, y, t)
    rtree = index.Index(properties=p)

    # insert trajectories into r-tree
    for traj in [traj1, traj2, traj3]:
        for i, node in enumerate(traj.nodes):
            rtree.insert(
                traj.id,  # id
                (node.x, node.y, node.t, node.x, node.y, node.t),  # Bbox (point)
                obj=(node.id, traj.id)  # we store the node ID & trajectory ID
            )

    # create query parameters
    params = {
        "t1": 0,  
        "t2": 3,  
        "eps": 0.5,  
        "linesMin": 2,  
        "origin": traj1,  
        "trajectories": [traj1, traj2, traj3]  
    }

    query = ClusterQuery(params)
    result = query.run(rtree)

    print("\nCluster Query Results:")
    print(f"Number of similar trajectories found: {len(result)}")
    print("\nSimilar trajectories:")
    for traj in result:
        print(f"Trajectory {traj.id}:")
        for node in traj.nodes:
            print(f"  Point: ({node.x}, {node.y}, {node.t})")