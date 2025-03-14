import os
import sys
import math

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
from rangeQuery import RangeQuery as RQ
from similarityQuery import SimilarityQuery as SQ
from Util import ParamUtil
import testFunctions as tF
import Trajectory as T
import Node as N
import Plot

def testSinglePlot():
    p1 = N.Node(1, 2.0, 3.0, 1.0)
    p2 = N.Node(2, 5.0, 10.0, 2.0)
    traj = T.Trajectory(1, [p1, p2])
    Plot.plotTrajectory(traj)

def testMultiPlot():
    p1 = N.Node(1, 2.0, 3.0, 1.0)
    p2 = N.Node(2, 3.5, 6.0, 2.0)
    p3 = N.Node(3, 5.0, 10.0, 2.0)
    traj1 = T.Trajectory(1, [p1, p2, p3])
    traj2 = T.Trajectory(2, [p1, p3])
    traj_list = [traj1, traj2]
    Plot.plotTrajectories(traj_list)

def testRangeQueryPlot(trajectories, rtree):
    # # Define the trajectories
    # p1 = N.Node(1, 2.0, 3.0, 1.0)
    # p2 = N.Node(2, 3.5, 6.0, 2.0)
    # p3 = N.Node(3, 5.0, 10.0, 2.0)
    # p4 = N.Node(4, 12.0, 13.0, 4.0)
    # p5 = N.Node(5, 15.0, 20.0, 4.0)
    # traj1 = T.Trajectory(1, [p1, p2, p3])
    # traj2 = T.Trajectory(2, [p1, p3])
    # traj3 = T.Trajectory(3, [p4, p5])
    # traj_list = [traj1, traj2, traj3]

    # Define the query
    query = RQ({"x1":-730000.0, "x2":-727000.0, "y1":4548000.0, "y2":4557000.0, "t1":-1000000000000000000.0, "t2":1500000000000000000.0})
    
    # Run the function to be tested
    Plot.plotRangeQuery(trajectories, query, rtree)

def testSimilarityQueryPlot(trajectories, rtree):
    paramUtil = ParamUtil(rtree, trajectories, delta = 50000000000000000)
    params = paramUtil.similarityParams(rtree, delta = 50000000000000000)
    query = SQ(params)
    Plot.plotSimilarity(trajectories, query, rtree)

def testPlotting(trajectories, rtree):
    testSinglePlot()
    testMultiPlot()
    testRangeQueryPlot(trajectories, rtree)
    testSimilarityQueryPlot(trajectories, rtree)