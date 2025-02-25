class Trajectory:
    # ID of the trajectory
    # All nodes in the trajectory implicitly assume this ID
    id: int

    # A list of all the nodes in the same trajectory
    nodes = []

    def __init__(self, id, nodes):
        # ID is assumed to be unique across all trajectories
        self.id = id

        # The parameter is assumed to be a list of 'Node' elements
        # IMPORTANT: This list is assumed to be totally ordered!
        self.nodes = nodes

    def __str__(self):
        out = "\n"

        # Get the string representation of each point in the trajectory
        for i in range(len(self.trajectory)):
            out += self.trajectory[i].__str__()
            print("out: " + out)

        # Return the full trajectory
        return f"Trajectory: \n[{out}]\n"
    
    def pointCount(self):
        return len(self.trajectory)