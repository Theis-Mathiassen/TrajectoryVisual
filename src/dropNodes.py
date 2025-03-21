from rtree import index
from src.Trajectory import Trajectory
from src.Node import Node
from tqdm import tqdm

def getNodes(trajectories):
    nodes = []
    for count, trajectory in enumerate(trajectories):
        for node in trajectory.nodes[1:-1]:
            nodes.append((node, count)) #Also stores index of associated trajectory, so we can easily find later
    return nodes
    

def dropNodes(rtree, trajectories, compression_rate, amount_to_drop = 0):
    # Drops a percentage of nodes, alternatively can drop an amount of nodes

    # Gets all nodes except first and last one, and sorts based on score
    nodes = getNodes(trajectories)
    sorted_nodes = sorted(nodes, key=lambda node: node[0].score)

    if(amount_to_drop <= 0):
        total_nodes = sum([len(x.nodes) for x in trajectories])
        amount_to_drop = total_nodes - round(total_nodes * compression_rate)
    

    # Special case for if there are so few nodes we cannot drop enough
    amount_to_drop = min(amount_to_drop, len(sorted_nodes))

    print("Dropping nodes..")
    for i in tqdm(range(amount_to_drop)):
        #Drop a single node
        (node, trajectory_index) = sorted_nodes[i]
        rtree.delete(node.id, (node.x, node.y, node.t, node.x, node.y, node.t))

        node_id = node.id

        # Find for associated trajectory
        for index, node in enumerate(trajectories[trajectory_index].nodes):
            if node.id == node_id:
                trajectories[trajectory_index].nodes.pop(index)