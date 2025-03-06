import sys
sys.path.append("src")
from Util import Node, Trajectory, lcss

n1 = Node(1, 1, 1, 1)
n2 = Node(2, 2, 2, 2)
n3 = Node(3, 3, 3, 3)
n4 = Node(4, 4, 4, 4)

t1 = Trajectory(1, [n1, n2, n3, n4])

n5 = Node(1, 1, 1, 1)
n6 = Node(2, 3, 3, 3)
n7 = Node(3, 5, 5, 5)
n8 = Node(4, 7, 7, 7)

t2 = Trajectory(1, [n5, n6, n7, n8])

result = lcss(2, 2, t1, t2)
print(result)