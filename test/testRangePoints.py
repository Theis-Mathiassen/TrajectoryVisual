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
import rangeQuery
import load
from Util import lonLatToMetric

x_min, y_min = lonLatToMetric(-8.645994,41.141412)
x_max, y_max = lonLatToMetric(-8.578521,41.178888)
params_1winner = {
    "x1": x_min,
    "y1": y_min,
    "t1": 1372616858,
    "x2": x_max,
    "y2": y_max,
    "t2": 1372738595,
    "flag": 1
}

params_divide1 = {
    "x1": x_min,
    "y1": y_min,
    "t1": 1372616858,
    "x2": x_max,
    "y2": y_max,
    "t2": 1372738595,
    "flag": 2
}

params_dividem = {
    "x1": x_min,
    "y1": y_min,
    "t1": 1372616858,
    "x2": x_max,
    "y2": y_max,
    "t2": 1372738595,
    "flag": 3
}


params_gradient = {
    "x1": x_min,
    "y1": y_min,
    "t1": 1372616858,
    "x2": x_max,
    "y2": y_max,
    "t2": 1372738595,
    "flag": 4
}

query_winner = rangeQuery.RangeQuery(params_1winner)
query_divide1 = rangeQuery.RangeQuery(params_divide1)
query_dividem = rangeQuery.RangeQuery(params_dividem)
query_gradient = rangeQuery.RangeQuery(params_gradient)

rTree, trajectories = load.build_Rtree("trimmed_small_train.csv")

qw_out = query_winner.run2(rTree)
query_winner.distribute(trajectories, qw_out)

qw_out = query_divide1.run2(rTree)
query_divide1.distribute(trajectories, qw_out)

qw_out = query_dividem.run2(rTree)
query_dividem.distribute(trajectories, qw_out)

#qw_out = query_gradient.run2(rTree)
#query_gradient.distribute(trajectories, qw_out)

for t in trajectories : 
    for n in t.nodes : 
        if n.score > 0 : 
            print(f"Node {n.id} has {n.score} points")



