import os
import sys

# Find the absolute path of the project directory
absolute_path = os.path.dirname(__file__)

# Define the relative paths of directories to import from
relative_path_src = "../src"
relative_path_test = "../test"

# Create the full path for these directories
full_path_src = os.path.join(absolute_path, relative_path_src)
full_path_test = os.path.join(absolute_path, relative_path_test)

# Append them to the path variable of the system
sys.path.append(full_path_src)
sys.path.append(full_path_test)

# All remaining imports
import load
from src.Util import ParamUtil
import testNode as tP
import testTrajectory as tT
import testPlot as tPP

def testMain():
    # load.load_Tdrive()
    Rtree_, Trajectories = load.build_Rtree("trimmed_small_train.csv", "simplified_Tdrive")

    # Initialize params
    paramUtil = ParamUtil(Rtree_, Trajectories, delta=5000000000.0)

    # Run all Node testing
    tP.testNodeClass()

    # Run all trajectory testing
    tT.testTrajectoryClass()

    # Run all plot testing
    tPP.testPlotting(Trajectories, Rtree_, paramUtil)

if __name__=="__main__":
    testMain()