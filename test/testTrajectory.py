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
import testFunctions as tF
import Node as P
import Trajectory as T

def testTrajectoryCreation():
    # Create a trajectory
    Node1 = P.Node(1, 3.0, 4.0, 1.0)
    Node2 = P.Node(2, 3.0, 4.0, 2.0)
    Nodes = [Node1, Node2]
    traj = T.Trajectory(1, Nodes)

    # Assert that it's ID field is initialized correctly
    tF.printAndAssertEQ(traj.id, 1)

def testTrajectoryClass():
    # Run all test functions for the trajectory class
    testTrajectoryCreation()