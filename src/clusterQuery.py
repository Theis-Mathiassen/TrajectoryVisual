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
        """Distribute points based on spatial proximity and directional similarity."""
        if not trajectories:
            return

        def calculate_direction(p1, p2):
            """Calculate direction vector between two points."""
            return np.array([p2.x - p1.x, p2.y - p1.y])

        def direction_similarity(dir1, dir2):
            """Calculate similarity between two direction vectors using dot product."""
            norm1 = np.linalg.norm(dir1)
            norm2 = np.linalg.norm(dir2)
            if norm1 == 0 or norm2 == 0:
                return 0
            # Normalize and compute dot product
            cos_angle = np.dot(dir1, dir2) / (norm1 * norm2)
            # Convert to a score between 0 and 1
            return (cos_angle + 1) / 2

        def spatial_proximity(p1, p2):
            """Calculate spatial proximity score between two points."""
            dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            # Convert distance to a score between 0 and 1 using exponential decay
            return np.exp(-dist / self.eps)

        # For each trajectory in the cluster
        for trajectory in trajectories:
            # Skip if trajectory has less than 2 points (needed for direction)
            if len(trajectory.nodes) < 2:
                continue

            # Calculate scores for each node
            for i in range(len(trajectory.nodes)):
                node = trajectory.nodes[i]
                spatial_score = 0
                direction_score = 0

                # Calculate spatial score
                for origin_node in self.origin.nodes:
                    spatial_score += spatial_proximity(node, origin_node)
                spatial_score /= len(self.origin.nodes)  # Normalize

                # Calculate direction score if not at the last point
                if i < len(trajectory.nodes) - 1:
                    traj_dir = calculate_direction(node, trajectory.nodes[i + 1])
                    # Compare with each segment in origin trajectory
                    dir_scores = []
                    for j in range(len(self.origin.nodes) - 1):
                        origin_dir = calculate_direction(self.origin.nodes[j], self.origin.nodes[j + 1])
                        dir_scores.append(direction_similarity(traj_dir, origin_dir))
                    direction_score = max(dir_scores) if dir_scores else 0

                # Combine scores (equal weights for simplicity)
                combined_score = (spatial_score + direction_score) / 2
                # Scale the score and add to node
                node.score += int(combined_score * 10)  # Scale to make scores more meaningful

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

    # Create R-tree index
    p = index.Property()
    p.dimension = 3  # 3D index (x, y, t)
    rtree = index.Index(properties=p)

    # Insert trajectories into R-tree
    for traj in [traj1, traj2, traj3]:
        for i, node in enumerate(traj.nodes):
            rtree.insert(
                traj.id,  # ID
                (node.x, node.y, node.t, node.x, node.y, node.t),  # Bbox (point)
                obj=(node.id, traj.id)  # Store node ID and trajectory ID
            )

    # Create query parameters
    params = {
        "t1": 0,  # Start time
        "t2": 3,  # End time
        "eps": 0.5,  # Maximum distance for clustering
        "linesMin": 2,  # Minimum number of lines in a cluster
        "origin": traj1,  # Use trajectory 1 as query trajectory
        "trajectories": [traj1, traj2, traj3]  # All available trajectories
    }

    # Create and run cluster query
    query = ClusterQuery(params)
    result = query.run(rtree)

    # Distribute scores
    query.distribute(result, None)  # matches parameter not used in our implementation

    print("\nCluster Query Results:")
    print(f"Number of similar trajectories found: {len(result)}")
    print("\nSimilar trajectories with scores:")
    for traj in result:
        print(f"\nTrajectory {traj.id}:")
        for node in traj.nodes:
            print(f"  Point: ({node.x}, {node.y}, {node.t}) - Score: {node.score}")