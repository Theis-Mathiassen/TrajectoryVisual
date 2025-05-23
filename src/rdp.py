# Based on https://github.com/fhirschmann/rdp/blob/master/rdp/__init__.py
# Adapted to fit our datastructures
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import numba as nb
import pickle
import os


def pldist(point, start, end):
    """
    Calculates the distance from ``point`` to the line given
    by the points ``start`` and ``end``.

    :param point: a point
    :type point: numpy array
    :param start: a point of the line
    :type start: numpy array
    :param end: another point of the line
    :type end: numpy array
    """
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
            np.abs(np.linalg.norm(np.cross(end - start, start - point))),
            np.linalg.norm(end - start))


def rdpMask(M, epsilon, dist=pldist):
    """
    Produces a mask of indices to keep based on the Ramer-Douglas-Peucker algorithm

    :param M: an array
    :type M: numpy array
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param dist: distance function
    :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
    """

    start_index = 0
    last_index = len(M) - 1

    stk = []
    stk.append([start_index, last_index])
    global_start_index = start_index
    indices = np.ones(last_index - start_index + 1, dtype=bool)

    while stk:
        start_index, last_index = stk.pop()

        dmax = 0.0
        index = start_index

        for i in range(index + 1, last_index):
            if indices[i - global_start_index]:
                d = dist(M[i], M[start_index], M[last_index])
                if d > dmax:
                    index = i
                    dmax = d

        if dmax > epsilon:
            stk.append([start_index, index])
            stk.append([index, last_index])
        else:
            for i in range(start_index + 1, last_index):
                indices[i - global_start_index] = False

    return indices


def extractNodes(trajectory):
    """
    Takes a trajectory and returns a list of nodes x and y coordinates in a numpy array

    :param trajectory: a trajectory
    :type trajectory: :class:`Trajectory`
    """

    return np.array([(node.x, node.y) for node in trajectory.nodes])

def rdpFilterTrajectories(trajectories, epsilon, score = 10, dist=pldist):
    """
    Takes a list of trajecoties, and applies Ramer-Douglas-Peucker to find nodes to score

    :param trajectories: list of trajectories
    :type trajectoties: :class:`Trajectory` array
    :param epsilon: epsilon in the rdp algorithm
    :type epsilon: float
    :param score: score to give retained nodes
    :type score: int
    :param dist: distance function
    :type dist: function with signature ``f(point, start, end)`` -- see :func:`rdp.pldist`
    """
    for trajectoryIndex, trajectory in enumerate(trajectories):
        nodes = extractNodes(trajectory) # Convert to numpy array
        mask = rdpMask(nodes, epsilon, dist)

        nodeIndicies = np.where(mask)[0]

        # Go through each node kept and award points
        for nodeIndex in nodeIndicies:
            trajectories[trajectoryIndex].nodes[nodeIndex].score += score # Currently we also give points to start and end points
   
   
        
def rdpMaskTrajectories(trajectories : dict, epsilon, dist=pldist):
    for trajectoryIndex in tqdm(trajectories.keys(), desc="RDP mask calculation"):
        nodes = extractNodes(trajectories[trajectoryIndex]) # Convert to numpy array
        mask = rdpMask(nodes, epsilon, dist)
        mask[0] = True
        mask[-1] = True
        trajectories[trajectoryIndex].nodes[~mask] = ma.masked
        


if __name__ == "__main__":
    path = os.getcwd()
    
    with open(os.path.join(path, 'rdpTrajectories.pkl'), 'rb') as file:
        trajectoriesWithRdpMask = pickle.load(file)
    
    totalNodes = 0
    totalUnmaskedNodes = 0
    
    for trajectoryidx, trajectory in trajectoriesWithRdpMask.items():
        totalNodes += trajectory.pointCount()
        totalUnmaskedNodes += trajectory.unmaskedCount()
        
    print(str.join("total: ", str(totalNodes), "\n", "unmasked: ", str(totalUnmaskedNodes), "\n", "ratio: ", str(totalUnmaskedNodes / totalNodes)))
    
    """print("\nTesting RDP...\n")
    arr = np.array([1, 1, 2, 2, 3, 3, 4, 4]).reshape(4, 2)

    print(arr)

    mask = rdpMask(arr, epsilon=0.5)
    print(mask)

    filteredArr = arr[mask]
    print(filteredArr)

    print("\nTesting RDP on Trajectories...\n")

    import random
    from Node import Node
    from Trajectory import Trajectory

    nodes = []
    for i in range(1, 6):
        nodes.append(Node(i, random.randint(i, i*3), random.randint(i, i*3), i))
    
    trajectory = Trajectory(1, nodes)

    # Use actual algorithm
    rdpFilterTrajectories([trajectory], 3)

    # Mimic algo

    nodes = extractNodes(trajectory)

    print("Generated nodes: ", nodes)


    mask = rdpMask(nodes, 3)

    print("Mask: ", mask)

    indicies = np.where(mask)[0]

    print("Indicies to score: ", indicies)

    for i in range(len(mask)):
        print("Node index: ", i, "\nScore: ", trajectory.nodes[i].score, "\n")"""
