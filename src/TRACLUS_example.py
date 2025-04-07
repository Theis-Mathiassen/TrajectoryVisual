import numpy as np
from TRACLUS import traclus
from src.load import load_Tdrive, build_Rtree

def convert_trajectories_to_numpy(trajectories):
    """Convert Trajectory objects to numpy arrays for TRACLUS"""
    numpy_trajectories = []
    for traj in trajectories:
        # extract x,y coordinates from nodes
        coords = np.array([[node.x, node.y] for node in traj.nodes])
        numpy_trajectories.append(coords)
    return numpy_trajectories

def main():
    print("loading T-drive dataset...")
    load_Tdrive("trimmed_small_train.csv")
    
    rtree, trajectories = build_Rtree("trimmed_small_train.csv", "test")
    
    numpy_trajectories = convert_trajectories_to_numpy(trajectories)
    
    print(f"\nDataset Statistics:")
    print(f"Number of trajectories: {len(numpy_trajectories)}")
    print(f"Average points per trajectory: {np.mean([len(traj) for traj in numpy_trajectories]):.2f}")
    
    # Apply TRACLUS clustering
    print("\nApplying TRACLUS clustering...")
    partitions, segments, dist_matrix, clusters, cluster_assignments, representative_trajectories = traclus(
        numpy_trajectories,
        max_eps=0.1,  
        min_samples=5,  
        directional=True,  
        use_segments=True,  
        mdl_weights=[1, 1, 1],  
        d_weights=[1, 1, 1],  
        progress_bar=True
    )
    
    # print clustering results
    print("\nClustering Results:")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Number of segments: {len(segments)}")
    print(f"Number of representative trajectories: {len(representative_trajectories)}")
    
    # print details of first 3 clusters
    print("\nFirst 3 clusters details:")
    for i, cluster in enumerate(clusters[:3]):
        print(f"\nCluster {i}:")
        print(f"Number of segments: {len(cluster)}")
        if len(representative_trajectories) > i:
            print(f"Representative traj points: {len(representative_trajectories[i])}")
            print(f"First few points of representative traj:")
            print(representative_trajectories[i][:3])

if __name__ == "__main__":
    main()