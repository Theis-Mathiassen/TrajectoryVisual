import os
import sys

# Find the absolute path of the project directory
absolute_path = os.path.dirname(__file__)

# Define the relative paths of directories to import from
relative_path_src = "../src"
relative_path_test = "../test"
relative_path_main = ".."

# Create the full path for these directories
full_path_src = os.path.join(absolute_path, relative_path_src)
full_path_test = os.path.join(absolute_path, relative_path_test)
full_path_main = os.path.join(absolute_path, relative_path_main)

# Append them to the path variable of the system
sys.path.append(full_path_src)
sys.path.append(full_path_test)
sys.path.append(full_path_main)

from Util import Node, Trajectory, lcss
import numpy as np
import numpy.ma as ma

n1 = Node(1, 1, 1, 1)
n2 = Node(2, 2, 2, 2)
n3 = Node(3, 3, 3, 3)
n4 = Node(4, 4, 4, 4)

t1 = Trajectory(1, ma.copy([n1, n2, n3, n4]))

n5 = Node(1, 1, 1, 1)
n6 = Node(2, 3, 3, 3)
n7 = Node(3, 5, 5, 5)
n8 = Node(4, 7, 7, 7)

t2 = Trajectory(1, ma.copy([n5, n6, n7, n8]))

result = lcss(2, 2, t1, t2)
print(result)