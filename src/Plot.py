import numpy as np
import matplotlib.pyplot as plt
from Trajectory import Trajectory as T
import math
from Node import Node as N
from Node import NodeDiff
from rangeQuery import RangeQuery as RQ
from similarityQuery import SimilarityQuery as SQ
from knnQuery import KnnQuery as KNN
#from clusterQuery import ClusterQuery as CQ
from rtree import Index
from numpy import ma
import contextily as cx
import geopandas as gpd
import tilemapbase
import pandas as pd
import Util

def trajectoryPlotting(traj: T):
        xs = []
        ys = []

        for elem in traj.nodes:
            xs.append(elem.x)
            ys.append(elem.y)
        
        plt.plot(xs, ys)

def plotTrajectory(traj: T):
    # Plots a single trajectory.
    # Input is assumed to be a 'Trajectory' instance.
    trajectoryPlotting(traj)
    plt.show()

def plotTrajectories(traj: list[T]):
    # Plots multiple trajectories.
    # Input is assumed to be a list of 'Trajectory' instances.
    for Trajectory in traj:
        trajectoryPlotting(Trajectory)
    plt.show()

def plottingSetup(query, rtree: Index):
    # Run the query
    q_traj = query.run(rtree)

    # Extract trajectory IDs
    ids = []
    for i in range(len(q_traj)):
        ids.append(q_traj[i].id)
    
    return q_traj, ids

def trajCoordSplit(trajectory: list[T]):
    xs = []
    ys = []
    
    # Create lists with the x and y coordinates of each point from a trajectory
    for elem in trajectory.nodes.data:
        xs.append(elem.x)
        ys.append(elem.y)
    
    return xs, ys

def trajCoordSplitMask(trajectory: list[T]):
    xs = []
    ys = []

    mask = ma.getmaskarray(trajectory.nodes)
    # Create lists with the x and y coordinates of each point from a trajectory
    c = 0
    for elem in trajectory.nodes.data:
        print(mask.data[c])
        if not mask[c] :
            xs.append(elem.x)
            ys.append(elem.y)
        c += 1
    
    return xs, ys

def plotRangeQuery(trajectories: list[T], query: RQ, rtree: Index):
    # Plots a query range
    # Plots all trajectories that intersect the query range as solid lines
    # All other trajectories are plotted with dashed lines

    _, ids = plottingSetup(query, rtree)

    # Loop over all trajectories in the dataset
    for Trajectory in trajectories:
        xs, ys = trajCoordSplit(Trajectory)

        # Differentiate the plot based on whether or not it was found by the query
        if Trajectory.id in ids:
            plt.plot(xs, ys, linestyle='solid')
        else:
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
    
    q_traj, ids = plottingSetup(query, rtree)

    # Setup variables for distance tracking and node memoization
    min_dist = math.inf
    q_traj_min_node = None
    res_traj_min_node = None
    for traj in q_traj:
        for node in traj.nodes:
            for q_node in query.trajectory.nodes:
                delta = np.linalg.norm([(q_node.x - node.x), (q_node.y - node.y)])
                if delta < min_dist:
                    min_dist = delta
                    if traj.id != query.trajectory.id:
                        res_traj_min_node = node
                        q_traj_min_node = q_node

    # Check if similar nodes have been found
    if res_traj_min_node != None and q_traj_min_node != None:

        # Plot closest point
        plt.scatter(res_traj_min_node.x, res_traj_min_node.y)

        # Plot a dotted line to this point from closest vantage within origin trajectory
        plt.plot([res_traj_min_node.x, q_traj_min_node.x],[res_traj_min_node.y, q_traj_min_node.y], linestyle='dotted')

        plt.show()

    # Loop over all trajectories in the dataset
    for Trajectory in trajectories:
        xs, ys = trajCoordSplit(Trajectory)
        
        if query.trajectory.id == Trajectory.id:
            plt.plot(xs, ys, linestyle='solid')
        elif Trajectory.id in ids:
            plt.plot(xs, ys, linestyle='solid', alpha=0.35)
        else:
            plt.plot(xs, ys, linestyle='dashed', alpha= 0.1)

    plt.show()

def plotkNNQuery(trajectories: list[T], query: KNN, rtree):
    # Plots a kNN query
    # Plots the trajectory in the query as a solid line
    # Plots all other trajectories returned by the query as a dashed line
    # Plots any trajectory not returned by the query as a dotted and translucent line

    _, ids = plottingSetup(query, rtree)

    # Loop over all trajectories in the dataset
    for Trajectory in trajectories:
        xs, ys = trajCoordSplit(Trajectory)
        
        # Do the differential plotting
        if query.trajectory.id == Trajectory.id:
            plt.plot(xs, ys, linestyle='solid', alpha=1.0)
        elif Trajectory.id in ids:
            plt.plot(xs, ys, linestyle='dashed', alpha=1.0)
        else:
            plt.plot(xs, ys, linestyle='dotted', alpha=0.1)

    plt.show()

#def plotClusterQuery(trajectories: list[T], query: CQ, rtree):
    # Needs refactoring of the abstract class before this works
 #   pass

def plotTrajectoryDiffFromMain(trajectories: dict[int, T]):
    counter = 0
    for traj in trajectories:
        if len(trajectories[traj].nodes) > 10:
            count = 0
            mask = ma.getmaskarray(trajectories[traj].nodes)
            for m in mask:
                if m:
                    count += 1
        if 10 > count > 5:
            if counter < 2:
                trajectories[traj].nodes = Util.ConvertAllMetricToLonLat(trajectories[traj])
                plotTrajectoryDiff(trajectories[traj])
                counter += 1
                break
    counter = 0
    for traj in trajectories:
        if len(trajectories[traj].nodes) > 10:
            count = 0
            mask = ma.getmaskarray(trajectories[traj].nodes)
            for m in mask:
                if not m:
                    count += 1
        if 10 > count > 5:
            if counter < 2:
                trajectories[traj].nodes = Util.ConvertAllMetricToLonLat(trajectories[traj])
                plotTrajectoryDiff(trajectories[traj])
                counter += 1
                break
            


def plotTrajectoryDiff(trajectory: T, counter, mask):
    # Plots a trajectory with the differences highlighted
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,8))


    xs, ys = trajCoordSplit(trajectory)
    c = [c for c in range(1,len(xs)+1)]
    d = {'col1': c, 'geometry': [(x,y) for x,y in zip(xs, ys)]}
    df = gpd.GeoDataFrame(d, crs="EPSG:4326", geometry=gpd.points_from_xy(xs, ys))
    # Plot the trajectory
    ax1.set_title("Original Trajectory")
    ax1.plot(xs,ys, color="navy", linestyle="dashed", alpha=0.3)
    ax1.scatter(xs, ys, marker="o", color="navy", s=20)


    # Plot the differences
    xss, yss = trajCoordSplitMask(trajectory)
    ax2.set_title("Simplified Trajectory")
    ax2.plot(xss,yss, color="red", linestyle="dashed", alpha=0.3)
    ax2.scatter(xss, yss, color="red", marker="^", s=20)

    BBox = (min(xs), max(xs), min(ys), max(ys))

    tilemapbase.start_logging()
    tilemapbase.init(create=True)
    #extent = tilemapbase.extent_from_frame(df, buffer = 25)   
    #plotter = tilemapbase.Plotter(extent, tilemapbase.tiles.build_OSM(), width=1000)
    #plotter.plot(ax1)
    for ax in (ax1, ax2):
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        #ax.grid(True)
        ax.set_xlim(BBox[0]-0.001, BBox[1]+0.001)
        ax.set_ylim(BBox[2]-0.002, BBox[3]+0.002)
        cx.add_basemap(ax, crs=df.crs, source=cx.providers.OpenStreetMap.Mapnik)
    if mask:
        savepath = f"TrajectoryDiffTrueMask_{counter}.pdf"
    else:
        savepath = f"TrajectoryDiffFalseMask_{counter}.pdf"
    plt.savefig(savepath, format="pdf", bbox_inches='tight')
    #plt.show()