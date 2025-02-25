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

def testPointCreation():
    # Create a point
    point1 = P.Point(1, 3.0, 4.0, 1.0)

    # Assert that all fields are initialized correctly
    tF.printAndAssertEQ(point1.x, 3.0)
    tF.printAndAssertEQ(point1.y, 4.0)
    tF.printAndAssertEQ(point1.t, 1.0)

def testDistanceCalc2D():
    point = P.Point(1, 3.0, 4.0, 1.0)
    tF.printAndAssertEQ(point.distanceEuclidean2D(), 5.0)

def testDistanceCalc3D():
    point = P.Point(1, 3.0, 4.0, 1.0)
    tF.printAndAssertEQ(point.distanceEuclidean3D(), math.sqrt(26.0))

def testCosineCalc():
    point = P.Point(1, 3.0, 4.0, 1.0)
    tF.printAndAssertEQ(point.pointCosine(), (3.0/5.0))

def testSineCalc():
    point = P.Point(1, 3.0, 4.0, 1.0)
    tF.printAndAssertEQ(point.pointSine(), (4.0/5.0))

def testPointDiff():
    point1 = P.Point(1, 3.0, 4.0, 1.0)
    point2 = P.Point(2, 5.0, 10.0, 2.0)
    tF.printAndAssertEQ(P.pointDiff(point1, point2)[0], -2.0)
    tF.printAndAssertEQ(P.pointDiff(point1, point2)[1], -6.0)


def testPointClass():
    # Run all test functions for the point class
    testPointCreation()
    testDistanceCalc2D()
    testDistanceCalc3D()
    testCosineCalc()
    testSineCalc()
    testPointDiff()