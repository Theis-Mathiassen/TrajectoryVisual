import math

class Point:
    # id: int

    # 2D coordinates
    x: float
    y: float

    # Timestamp (can be used as a third coordinate for 3D computations)
    t: float

    def __init__(self, id, x, y, t):
        # Basic initialization.
        # All values are passed to the object at creation
        # self.id = id
        self.x = x
        self.y = y
        self.t = t

    def __str__(self):
        return f"x: {self.x}\ny: {self.y}\ntime: {self.t}\n"
    
    def distanceEuclidean2D(self):
        # Calculates the norm of the 2D-point vector for this point
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def distanceEuclidean3D(self):
        # Calculates the norm of the 3D-point vector, 
        # where the time field is interpreted as the third dimension.
        return math.sqrt(self.x * self.x + self.y * self.y + self.t * self.t)
    
    def pointCosine(self):
        # Calculates the cosine of the point vector
        return (self.x / self.distanceEuclidean2D())
    
    def pointSine(self):
        # Calculates the sine of the point vector
        return (self.y / self.distanceEuclidean2D())
    

def pointDiff(point1: Point, point2: Point):
    # Calculates the difference in the 2D euclidean coordinates between two points
    return ((point1.x - point2.x), (point1.y - point2.y))