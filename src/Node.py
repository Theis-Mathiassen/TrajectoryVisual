import math

class Node:
    score: int
    id: int

    # 2D coordinates
    x: float
    y: float

    # Timestamp (can be used as a third coordinate for 3D computations)
    t: float

    def __init__(self, id, x, y, t):
        # Basic initialization.
        # All values are passed to the object at creation
        self.id = id
        self.x = x
        self.y = y
        self.t = t
        self.score = 0

    def __str__(self):
        return f"x: {self.x}\ny: {self.y}\ntime: {self.t}\n"
    
    """ def distanceEuclidean2D(self):
        # Calculates the norm of the 2D-node vector for this node
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def distanceEuclidean3D(self):
        # Calculates the norm of the 3D-node vector, 
        # where the time field is interpreted as the third dimension.
        return math.sqrt(self.x * self.x + self.y * self.y + self.t * self.t)
    

    def NodeCosine(self):
        # Calculates the cosine of the node vector
        return (self.x / self.distanceEuclidean2D())
    
    def NodeSine(self):
        # Calculates the sine of the node vector
        return (self.y / self.distanceEuclidean2D()) """
    

def NodeDiff(node1: Node, node2: Node):
    # Calculates the difference in the 2D euclidean coordinates between two nodes
    return ((node1.x - node2.x), (node1.y - node2.y))