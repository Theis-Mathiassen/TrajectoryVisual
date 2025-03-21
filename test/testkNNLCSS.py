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

import testFunctions as tF
import Node as P
import Trajectory as T
import knnQuery
import load
from Util import lonLatToMetric, ParamUtil

x_min, y_min = lonLatToMetric(-8.645994,41.141412)
x_max, y_max = lonLatToMetric(-8.578521,41.178888)

rtree, trajectories = load.build_Rtree("trimmed_small_train.csv")

params = ParamUtil(rtree, trajectories, 1, 3, 1).knnParams(rtree, flag=2)

query = knnQuery.KnnQuery(params)

q_out = query.run(rtree=rtree)
query.distribute(trajectories, q_out)