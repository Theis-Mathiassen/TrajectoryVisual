from rtree import index
from src.Trajectory import Trajectory
from src.Node import Node
from tqdm import tqdm
import numpy.ma as ma
import numpy as np

def getNodes(trajectories):
    """ nodes = []
    for count, trajectory in enumerate(trajectories):
        for node in trajectory.nodes[1:-1]:
            nodes.append((node, count)) #Also stores index of associated trajectory, so we can easily find later """
    nodes = []
    for trajectory in trajectories.values():
        zipped = [[node, trajectory.id] for node in trajectory.nodes[1:-1]]
        nodes += zipped
        
    return nodes
    

def dropNodes(rtree, trajectories, compression_rate, amount_to_drop = 0):
    # Drops a percentage of nodes, alternatively can drop an amount of nodes

    # Gets all nodes except first and last one, and sorts based on score
    nodes = getNodes(trajectories)
    sorted_nodes = sorted(nodes, key=lambda node: node[0].score)

    if(amount_to_drop <= 0):
        total_nodes = sum([len(x.nodes) for x in trajectories.values()])
        amount_to_drop = total_nodes - round(total_nodes * compression_rate)
    
    
    # Special case for if there are so few nodes we cannot drop enough
    amount_to_drop = min(amount_to_drop, len(sorted_nodes))

    
    nodesToDrop = sorted_nodes[0 : amount_to_drop]
    nodesPerTrajectory = {}
    for node_id, trajectory_id in nodesToDrop:
        if trajectory_id not in nodesPerTrajectory:
            nodesPerTrajectory[trajectory_id] = []
        nodesPerTrajectory[trajectory_id].append(node_id)
    
    
    for trajectory_id in tqdm(nodesPerTrajectory.keys(), total=len(nodesPerTrajectory.keys()), desc="Dropping nodes"):
        trajectory = trajectories.get(trajectory_id)
        mask = np.zeros(len(trajectory.nodes))
        droppedNodes = nodesPerTrajectory[trajectory_id]
        for node in droppedNodes:
            mask[node.id] = 1
        trajectory.nodes = ma.array(trajectories[trajectory_id].nodes, mask = mask)
    
    """ for i in tqdm(range(amount_to_drop)):
        #Drop a single node
        (node, trajectory_index) = sorted_nodes[i]
        # rtree.delete(node.id, (node.x, node.y, node.t, node.x, node.y, node.t))

        node_id = node.id
        # Change to work with masked arrays
        # Find for associated trajectory
        for index, node in enumerate(trajectories[trajectory_index].nodes):
            if node.id == node_id:
                
                trajectories[trajectory_index].nodes = ma.array(trajectories[trajectory_index].nodes)
                #trajectories[trajectory_index].nodes.pop(index)
                break """
    return trajectories