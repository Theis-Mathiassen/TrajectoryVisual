import numpy as np
import matplotlib.pyplot as plt
from Trajectory import Trajectory as T
import math
from Node import Node as N
from rangeQuery import RangeQuery as RQ
from similarityQuery import SimilarityQuery as SQ
from rtree import Index

def plotTrajectory(traj: T):
    # Plots a single trajectory.
    # Input is assumed to be a 'Trajectory' instance.
    xs = []
    ys = []

    for elem in traj.nodes:
        xs.append(elem.x)
        ys.append(elem.y)

    plt.plot(xs, ys)
    plt.show()

def plotTrajectories(traj: list[T]):
    # Plots multiple trajectories.
    # Input is assumed to be a list of 'Trajectory' instances.
    for Trajectory in traj:
        xs = []
        ys = []

        for elem in Trajectory.nodes:
            xs.append(elem.x)
            ys.append(elem.y)

        plt.plot(xs, ys)
    plt.show()

def plotRangeQuery(trajectories: list[T], query: RQ, rtree: Index):
    # Plots a query range
    # Plots all trajectories that intersect the query range as solid lines
    # All other trajectories are plotted with dashed lines

    # Run the query
    q_traj = query.run(rtree)

    # Extract trajectory IDs
    ids = []
    for i in range(len(q_traj)):
        ids.append(q_traj[i].id)

    # Loop over all trajectories in the dataset
    for Trajectory in trajectories:
        xs = []
        ys = []
        
        # Create lists with the x and y coordinates of each point from a trajectory
        for elem in Trajectory.nodes:
            xs.append(elem.x)
            ys.append(elem.y)

        # Differentiate the plot based on whether or not it was found by the query
        try:
            ids.remove(Trajectory.id)
            plt.plot(xs, ys, linestyle='solid')
        except:
            plt.plot(xs, ys, linestyle='dashed', alpha=0.6)

    # Plot a rectangle thats equivalent to the range specified by the query
    x_range = [query.x1, query.x1, query.x2, query.x2]
    y_range = [query.y1, query.y2, query.y2, query.y1]
    plt.fill(x_range, y_range, alpha=0.3)

    plt.show()

def plotSimilarity(trajectories: list[T], query: SQ, rtree):
    # Plots a similarity query
    # Plots the trajectory in the query as a solid line
    # Plots all other trajectories returned by the query as a dashed line
    
    # Run the query
    q_traj = query.run(rtree)
    
    # Save the id of the trajectory being queried
    id = query.trajectory.id

    # Extract trajectory IDs
    ids = []
    for i in range(len(q_traj)):
        ids.append(q_traj[i].id)

    # Loop over all trajectories in the dataset
    for Trajectory in trajectories:
        xs = []
        ys = []

        # Create lists with the x and y coordinates of each point from a trajectory
        for elem in Trajectory.nodes:
            xs.append(elem.x)
            ys.append(elem.y)
        
        if id == Trajectory.id:
            plt.plot(xs, ys, linestyle='solid')
        try:
            ids.remove(Trajectory.id)
            plt.plot(xs, ys, linestyle='solid', alpha=0.7)
        except:
            plt.plot(xs, ys, linestyle='dashed', alpha= 0.45)

    plt.show()