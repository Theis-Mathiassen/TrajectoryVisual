import numpy as np 
from Node import Node
from Filter import Filter


class StayPointFilter(Filter):
    def __init__(self, consecutive_nodes_for_a_stop = 5, spatial_radius_m = 200, verbose = False):
        self.consecutive_nodes_for_a_stop = consecutive_nodes_for_a_stop
        self.spatial_radius_m = spatial_radius_m
        self.verbose = verbose

    def filterTrajectories(self, trajectories):
        # Function alters the input trajectory
        stayPointFilter(trajectories, self.consecutive_nodes_for_a_stop, self.spatial_radius_m, self.verbose)




def get_stay_locations(nodes, consecutive_nodes_for_a_stop = 5,  spatial_radius_m = 200):
    """
    Detects when trajectories have stopped and returns the locations of the stop
    """

    stay_locations = []
    stay_indexes = []   # Of the form [(start index, end index), (start index, end index), ...]

    count = 1

    current_spot = nodes[0]
    sum_location = [current_spot]


    for index in range(1, len(nodes)):

        current_node = nodes[index]

        distance_measure = np.linalg.norm(current_spot - current_node)

        if distance_measure < spatial_radius_m:    # if a stop
            count += 1                              # Increment count
            sum_location.append(current_node)       # Add to sum


        else: # If not a stop
            if count >= consecutive_nodes_for_a_stop:   # If enough nodes in stop to add to list
                # Add to list of stay locations
                median_Location = np.median(sum_location, axis=0)   # Get median location of the stop   
                stay_locations.append(median_Location)

                # Add to list of stay indexes
                stay_indexes.append((index - (count - 1), index))
            
            # Reset params
            count = 1
            sum_location = [current_node]
            current_spot = current_node
            

    return stay_locations, stay_indexes


def stayPointFilter(trajectories, consecutive_nodes_for_a_stop = 5,  spatial_radius_m = 200, verbose = False):
    """
    Detects when trajectories have stopped and filters those locations out to just include 1
    """
    deleteAmount = 0
    reinsertAmount = 0


    for trajectoryIndex, trajectory in enumerate(trajectories):
        nodes = extractNodes(trajectory)

        stay_locations, stay_indexes = get_stay_locations(nodes, consecutive_nodes_for_a_stop, spatial_radius_m)

        if verbose: # Collect data on how many deleted, and how many reinserted
            for (start, end) in stay_indexes:
                deleteAmount += (start - end) + 1
            reinsertAmount += len(stay_locations)

        newNodes = []

        # Create list of nodes to retain
        for i in range(len(stay_locations)):
            x_idx, y_idx = stay_locations[i]
            start, end = stay_indexes[i]

            t_min = trajectory.nodes[start].t
            t_max = trajectory.nodes[end].t

            node_id = trajectory.nodes[start].id # Copy the first id

            newNodes.append(Node(node_id, x_idx, y_idx, t_max - t_min)) # New node with average location


        # Delete the nodes
        filterMask = np.ones(len(nodes), dtype=bool)

        for (start, end) in stay_indexes:
            filterMask[start:end + 1] = False

        trajectory.nodes = [trajectory.nodes[i] for i in range(len(trajectory.nodes)) if filterMask[i]]
        # With numpy
        # trajectory.nodes = trajectory.nodes[filterMask]

        for node in newNodes:   # Append new nodes
            trajectory.nodes.append(node)

        trajectory.nodes = sorted(trajectory.nodes, key=lambda node: node.t)    # Sort by time

    if verbose:
        print(f"Removed {deleteAmount} nodes, reinserted {reinsertAmount} nodes")


def extractNodes(trajectory):
    """
    Takes a trajectory and returns a list of nodes x and y coordinates in a numpy array
    :param trajectory: a trajectory
    :type trajectory: :class:`Trajectory`
    """

    return np.array([(node.x, node.y) for node in trajectory.nodes])


if __name__ == "__main__":


    import random
    from Node import Node
    from Trajectory import Trajectory

    nodes = []
    amount = 10000
    for i in range(1, amount + 1):
        nodes.append(Node(i, random.randint(i, i+10), random.randint(i, i+10), i))

    trajectory = Trajectory(1, nodes)


    nodes = extractNodes(trajectory)

    stay_locations, stay_indexes = get_stay_locations(nodes, 5, 3)

    for i in range(len(stay_locations)):
        print("---Stays---")
        print(stay_locations[i], stay_indexes[i])

        for j in range(stay_indexes[i][0], stay_indexes[i][1] + 1):
            print(nodes[j])

    print("\n\n")
