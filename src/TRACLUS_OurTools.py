from src.TRACLUS import partition, partition2segments
import numpy as np
from tqdm import tqdm



def traclus_get_segments(trajectories, directional=True, use_segments=True, mdl_weights=[1,1,1], return_partitions = False):
    """
        Trajectory Clustering Algorithm - Get Paritions / Segments
    """
    # Ensure that the trajectories are a list of numpy arrays of shape (n, 2)
    if not isinstance(trajectories, list):
        raise TypeError("Trajectories must be a list")
    for trajectory in trajectories:
        if not isinstance(trajectory, np.ndarray):
            raise TypeError("Trajectories must be a list of numpy arrays")
        elif len(trajectory.shape) != 2:
            raise ValueError("Trajectories must be a list of numpy arrays of shape (n, 2)")
        elif trajectory.shape[1] != 2:
            raise ValueError("Trajectories must be a list of numpy arrays of shape (n, 2)")

    # Partition the trajectories

    partitions = []
    i = 0
    for trajectory in tqdm(trajectories, desc="Partitioning trajectories"):

        partitions.append(partition(trajectory, directional=directional, progress_bar=False, w_perpendicular=mdl_weights[0], w_angular=mdl_weights[2]))


    if return_partitions:
        return partitions

    # Get the segments for each partition
    segments = []
    if use_segments:
        for parts in tqdm(partitions, desc="Converting partitioned trajectories to segments"):
            segments += partition2segments(parts)
    else:
        segments = partitions

    return segments