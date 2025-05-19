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
import testFunctions as tF
import Node as P

def testNodeCreation():
    # Create a Node
    Node1 = P.Node(1, 3.0, 4.0, 1.0)

    # Assert that all fields are initialized correctly
    tF.printAndAssertEQ(Node1.x, 3.0)
    tF.printAndAssertEQ(Node1.y, 4.0)
    tF.printAndAssertEQ(Node1.t, 1.0)

def testDistanceCalc2D():
    Node = P.Node(1, 3.0, 4.0, 1.0)
    tF.printAndAssertEQ(Node.distanceEuclidean2D(), 5.0)

def testDistanceCalc3D():
    Node = P.Node(1, 3.0, 4.0, 1.0)
    tF.printAndAssertEQ(Node.distanceEuclidean3D(), math.sqrt(26.0))

def testCosineCalc():
    Node = P.Node(1, 3.0, 4.0, 1.0)
    tF.printAndAssertEQ(Node.NodeCosine(), (3.0/5.0))

def testSineCalc():
    Node = P.Node(1, 3.0, 4.0, 1.0)
    tF.printAndAssertEQ(Node.NodeSine(), (4.0/5.0))

def testNodeDiff():
    Node1 = P.Node(1, 3.0, 4.0, 1.0)
    Node2 = P.Node(2, 5.0, 10.0, 2.0)
    tF.printAndAssertEQ(P.NodeDiff(Node1, Node2)[0], -2.0)
    tF.printAndAssertEQ(P.NodeDiff(Node1, Node2)[1], -6.0)


def testNodeClass():
    # Run all test functions for the Node class
    testNodeCreation()
    #testDistanceCalc2D()
    #testDistanceCalc3D()
    #testCosineCalc()
    #testSineCalc()
    testNodeDiff()