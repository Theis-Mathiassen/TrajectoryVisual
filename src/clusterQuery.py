import os
import sys

# Find the absolute path of the project directory
absolute_path = os.path.dirname(__file__)

# Define the relative paths of directories to import from
relative_path_src = ".."
relative_path_test = "../src"

# Create the full path for these directories
full_path_src = os.path.join(absolute_path, relative_path_src)
full_path_test = os.path.join(absolute_path, relative_path_test)

# Append them to the path variable of the system
sys.path.append(full_path_src)
sys.path.append(full_path_test)


from src.Query import Query
from src.Trajectory import Trajectory
import numpy as np
from src.TRACLUS import traclus
from sklearn.cluster import OPTICS
from src.Node import Node
from rtree import index
from src.Util import euc_dist_diff_3d
from collections import defaultdict
from TRACLUS_OurTools import traclus_get_segments

class ClusterQuery(Query): 

    def __init__(self, params):
        super().__init__(params)
        self.t1 = params["t1"]  
        self.t2 = params["t2"]  # t1=start time, t2=end time
        self.eps = params["eps"]  # max distance for clustering
        self.min_lines = params["linesMin"]  # min number of lines in a cluster
        self.origin = params["origin"]  # the query trajectory
        self.hits = []  # stores hits. hit = an entry in R-tree that satisfies the search cond. (i.e. within time window)
        self.params = params
        self.returnCluster = False

    def _trajectory_to_numpy(self, trajectory):
        """Convert a Trajectory object to numpy array format required by TRACLUS."""
        return np.array([[node.x, node.y] for node in trajectory.nodes])

    def _filter_trajectories_by_time(self, trajectories, rtree):
        """Filter trajectories based on temporal constraints using R-tree."""
        # use float inf to ensure all trajectories are considered, regardless of their values. So we only filter by time
        hits = list(rtree.intersection((float('-inf'), float('-inf'), self.t1, 
                                        float('inf'), float('inf'), self.t2), objects=True))
        self.hits = hits  
        # a single hit is stored as: (node_id, trajectory_id)


        filtered = []
        seen_trajectories = set()
        for hit in hits:
            _, trajectory_id = hit.object
            if trajectory_id not in seen_trajectories:
                seen_trajectories.add(trajectory_id)
                matching_traj = next((t for t in trajectories.keys() if t == trajectory_id), None)
                if matching_traj:
                    filtered.append(matching_traj)
        
        return filtered
    
    def run(self, rtree):
        # get all trajectories with points in the time window, unless needs to return all clusters
        if self.returnCluster:
            trajectories = self.params["trajectories"]
        else: 
            trajectories = self._filter_trajectories_by_time(self.params["trajectories"], rtree)
        
        if not trajectories:
            return []

        # convert trajectories to numpy arrays for TRACLUS
        numpy_trajectories = [self._trajectory_to_numpy(t) for t in trajectories]
        origin_numpy = self._trajectory_to_numpy(self.origin)

        if not self.returnCluster: # If centered around an origin trajectory
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

        if self.returnCluster:  # If should instead "just" return the clusters
            # Since none are filtered out, they are in the same order. We then group them by values (Cluster ids)
            dict_for_clusters = defaultdict(list)
            for index, value in enumerate(cluster_assignments):
                dict_for_clusters[value].append(trajectories[index])

            return dict_for_clusters.values()  # Return groupings

        # find which cluster contains the query trajectory
        query_idx = len(numpy_trajectories) - 1  # last added trajectory is the query
        query_cluster = cluster_assignments[query_idx]

        # get trajectories in the same cluster as the query
        similar_trajectories = []
        for idx, cluster_id in enumerate(cluster_assignments):
            if cluster_id == query_cluster and idx != query_idx:
                similar_trajectories.append(trajectories[idx])

        return similar_trajectories

    def distribute(self, trajectories):
        """Distribute points based on cluster membership and spatial proximity."""
        if not trajectories:
            return

        def give_point(trajectory: Trajectory, node_id):
            for n in trajectory.nodes:
                if n.id == node_id:
                    n.score += 1

        # Calculate query center (using origin trajectory)
        center_x = self.params['x1'] + self.params['x2'] / 2
        center_y = self.params['y1'] + self.params['y2'] / 2
        center_t = self.params['t1'] + self.params['t2'] / 2
        """ center_x = np.mean([node.x for node in self.origin.nodes.data])
        center_y = np.mean([node.y for node in self.origin.nodes.data])
        center_t = np.mean([node.t for node in self.origin.nodes.data]) """
        q_bbox = [center_x, center_y, center_t]

        # Key = Trajectory id, value = (Node id, distance)
        point_dict = dict()

        # get the hits into correct format
        matches = [(n.object, n.bbox) for n in self.hits]

        for obj, bbox in matches:
            dist_current = euc_dist_diff_3d(bbox, q_bbox)

            if obj[0] in point_dict:
                dist_prev = point_dict.get(obj[0])[1]
                if dist_current <= dist_prev:
                    point_dict[obj[0]] = (obj[1], dist_current)
            else:
                point_dict[obj[0]] = (obj[1], dist_current)

        # distribute points to the closest nodes in each trajectory
        for key, value in point_dict.items():
            trajectories[key].nodes[value[0]].score += 1
            """ for t in trajectories.values():
                if t.id == key:
                    give_point(t, value[0]) """

    def distributeCluster(self, trajectories, scoreToAward = 1):
        # convert trajectories to numpy arrays for TRACLUS
        numpy_trajectories = [self._trajectory_to_numpy(t) for t in trajectories]

        partitions = traclus_get_segments(
            trajectories=numpy_trajectories,
            directional=True,
            use_segments=True,
            progress_bar=False,
            return_partitions=True
        )

        nodesToReward = dict()

        # Find node indicies for each trajectory. Has to be iterative as trajectory lengths vary
        for trajectoryIndex, (trajectory, partition) in enumerate(zip(numpy_trajectories, partitions)):
            mask = (trajectory[:, None] == partition).all(axis=2)
            indices = np.where(mask)[0]

            nodesToReward[trajectoryIndex] = indices

        # Award points
        for trajectoryIndex, nodeIndexes in nodesToReward.items():
            for nodeIndex in nodeIndexes:
                trajectories[trajectoryIndex].nodes[nodeIndex].score += scoreToAward


    def distributeCluster(self, trajectories, scoreToAward = 1):
        # convert trajectories to numpy arrays for TRACLUS
        numpy_trajectories = [self._trajectory_to_numpy(t) for t in trajectories]

        partitions = traclus_get_segments(
            trajectories=numpy_trajectories,
            directional=True,
            use_segments=True,
            progress_bar=False,
            return_partitions=True
        )

        nodesToReward = dict()

        # Find node indicies for each trajectory. Has to be iterative as trajectory lengths vary
        for trajectoryIndex, (trajectory, partition) in enumerate(zip(numpy_trajectories, partitions)):
            mask = (trajectory[:, None] == partition).all(axis=2)
            indices = np.where(mask)[0]

            nodesToReward[trajectoryIndex] = indices

        # Award points
        for trajectoryIndex, nodeIndexes in nodesToReward.items():
            for nodeIndex in nodeIndexes:
                trajectories[trajectoryIndex].nodes[nodeIndex].score += scoreToAward


if __name__ == "__main__":
    # Create sample trajectories for testing
    def create_sample_trajectory(id, points):
        nodes = [Node(i, x, y, t) for i, (x, y, t) in enumerate(points)]
        return Trajectory(id, nodes)

    # Create some sample trajectories
    # Trajectory 1 and 2 follow similar paths
    traj1 = create_sample_trajectory(1, [
        (0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)
    ])
    traj2 = create_sample_trajectory(2, [
        (0.1, 0.1, 0), (1.1, 1.1, 1), (2.1, 2.1, 2), (3.1, 3.1, 3)
    ])
    # Trajectory 3 follows a different path
    traj3 = create_sample_trajectory(3, [
        (0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3)
    ])

    # create R-tree index
    p = index.Property()
    p.dimension = 3  
    rtree = index.Index(properties=p)

    # insert trajectories into R-tree
    for traj in [traj1, traj2, traj3]:
        for i, node in enumerate(traj.nodes):
            rtree.insert(
                traj.id,  # ID
                (node.x, node.y, node.t, node.x, node.y, node.t),  # Bbox (point)
                obj=(node.id, traj.id)  
            )

    # query params
    params = {
        "t1": 0,  
        "t2": 3,  
        "eps": 0.5,  # max distance for clustering
        "linesMin": 2,  # min amount of lines in a cluster
        "origin": traj1,  # query trajectory
        "trajectories": [traj1, traj2, traj3] 
    }

    query = ClusterQuery(params)

    result = query.run(rtree)

    query.distribute(result)

    print("\nCluster Query Results:")
    print(f"Number of similar trajectories found: {len(result)}")
    print("\nSimilar trajectories with scores:")
    for traj in result:
        print(f"\nTrajectory {traj.id}:")
        for node in traj.nodes:
            print(f"  Point: ({node.x}, {node.y}, {node.t}) - Score: {node.score}")

    # Reset points


    nodes = [(0,0,0)]
    x = 0
    y = 0
    t = 0
    for i in range(10):
        t += 1
        x += 10
        nodes.append((x, y, t))
    
    for i in range(10):
        t += 1
        y += 10
        nodes.append((x, y, t))

    traj4 = create_sample_trajectory(4, nodes)

    query.distributeCluster(trajectories=[traj1, traj2, traj3, traj4])


    print("\nCluster Query Results:")
    print(f"Number of similar trajectories found: {len(result)}")
    print("\nSimilar trajectories with scores:")
    for traj in result:
        print(f"\nTrajectory {traj.id}:")
        for node in traj.nodes:
            print(f"  Point: ({node.x}, {node.y}, {node.t}) - Score: {node.score}")

    print("\n ----- Scoring for all nodes---- \n")

    for traj in [traj1, traj2, traj3, traj4]:
        print(f"\nTrajectory {traj.id}:")
        for node in traj.nodes:
            print(f"  Point: ({node.x}, {node.y}, {node.t}) - Score: {node.score}")



    query.returnCluster = True

    clusters = query.run(rtree)

    print("\n Clusters: \n")
    for index, cluster in enumerate(clusters):
        print(f"Cluster {index + 1}:")
        for trajectory in cluster:
            print(f"Trajectory {trajectory.id}")

        print("\n")