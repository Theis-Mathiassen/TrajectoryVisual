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
from knnQuery import KnnQuery as KNN
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

def testRangeQueryPlot(trajectories, rtree, paramUtil: ParamUtil):   
    # Dynamically defined query to test implementation across the board
    params = paramUtil.rangeParams(rtree)
    query = RQ(params)

    # Statically defined query for reproducability.
    params['x1'] += 0.05
    params['x2'] -= 0.05
    params['y1'] += 0.05
    params['y2'] -= 0.05
    query2 = RQ(params)


    # Run the function to be tested
    #Plot.plotRangeQuery(trajectories, query, rtree)

    #Plot.plotRangeQuery(trajectories, query2, rtree)

    Plot.plotWTA(trajectories, query2, rtree)
    Plot.plotSE(trajectories, query2, rtree)
    Plot.plotGP2(trajectories, query2, rtree)
    Plot.plotGP(trajectories, query2, rtree)

def testSimilarityQueryPlot(trajectories, rtree, paramUtil: ParamUtil):
    params = paramUtil.similarityParams(rtree)
    query = SQ(params)
    Plot.plotSimilarity(trajectories, query, rtree)

def testKNNQueryPlot(trajectories, rtree, paramUtil: ParamUtil):
    params = paramUtil.knnParams(rtree)
    query = KNN(params)
    Plot.plotkNNQuery(trajectories, query, rtree)

def testClusterQuery(trajectories, rtree, paramUtil: ParamUtil):
    pass

def testPlotting(trajectories, rtree, paramUtil):
    #testSinglePlot()
    #testMultiPlot()
    testRangeQueryPlot(trajectories, rtree, paramUtil)
    #testSimilarityQueryPlot(trajectories, rtree, paramUtil)
    testKNNQueryPlot(trajectories, rtree, paramUtil)