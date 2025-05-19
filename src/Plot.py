import numpy as np
import matplotlib.pyplot as plt
from Trajectory import Trajectory as T
import math
from Node import Node as N
from Node import NodeDiff
from rangeQuery import RangeQuery as RQ
from similarityQuery import SimilarityQuery as SQ
from knnQuery import KnnQuery as KNN
from clusterQuery import ClusterQuery as CQ
from rtree import Index
from Util import lonLatToMetric

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

def plottingSetup(trajectories, query, rtree: Index):
    # Run the query
    q_traj = query.run(rtree, trajectories)
    
    # Extract trajectory IDs
    ids = []
    for i in range(len(q_traj)):
        #print(q_traj[i])
        ids.append(q_traj[i][0])
    
    return q_traj, ids

def trajCoordSplit(trajectory: T):
    xs = []
    ys = []
    
    # Create lists with the x and y coordinates of each point from a trajectory
    for elem in trajectory.nodes:
        xs.append(elem.x)
        ys.append(elem.y)
    
    return xs, ys

def plotRangeQuery(trajectories: list[T], query: RQ, rtree: Index):
    # Plots a query range
    # Plots all trajectories that intersect the query range as solid lines
    # All other trajectories are plotted with dashed lines

    _, ids = plottingSetup(query, rtree)

    # Loop over all trajectories in the dataset
    for Trajectory in trajectories:
        lon, lat = trajCoordSplit(trajectories[Trajectory])
        xs, ys = [], []

        for (x, y) in zip(lon, lat):
            temp = lonLatToMetric(x, y)
            xs.append(temp[0])
            ys.append(temp[1])

        # Differentiate the plot based on whether or not it was found by the query
        if Trajectory in ids:
            plt.plot(xs, ys, linestyle='solid')
        else:
            plt.plot(xs, ys, linestyle='dashed', alpha=0.6)

    # Plot a rectangle thats equivalent to the range specified by the query
    x_range = [query.x1, query.x1, query.x2, query.x2]
    y_range = [query.y1, query.y2, query.y2, query.y1]
    plt.fill(x_range, y_range, alpha=0.3)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.show()

def plotSimilarity(trajectories: list[T], query: SQ, rtree):
    # Plots a similarity query
    # Plots the trajectory in the query as a solid line
    # Plots all other trajectories returned by the query as a dashed line
    
    q_traj, ids = plottingSetup(query, rtree)
    #print(query)
    # Setup variables for distance tracking and node memoization
    min_dist = math.inf
    q_traj_min_node = None
    res_traj_min_node = None
    for id in ids:
        for node in trajectories[id].nodes:
            for q_node in query.trajectory.nodes:
                delta = np.linalg.norm([(q_node.x - node.x), (q_node.y - node.y)])
                if delta <= min_dist and trajectories[id].id != query.trajectory.id:
                    min_dist = delta
                    res_traj_min_node = node
                    q_traj_min_node = q_node

    # Check if similar nodes have been found
    if res_traj_min_node != None and q_traj_min_node != None:
        res_traj_min_node.x, res_traj_min_node.y = lonLatToMetric(res_traj_min_node.x, res_traj_min_node.y)

        # Plot closest point
        plt.scatter(res_traj_min_node.x, res_traj_min_node.y)

        # Plot a dotted line to this point from closest vantage within origin trajectory
        plt.plot([res_traj_min_node.x, q_traj_min_node.x],[res_traj_min_node.y, q_traj_min_node.y], linestyle='dotted')

    # Loop over all trajectories in the dataset
    for Trajectory in trajectories:
        lon, lat = trajCoordSplit(trajectories[Trajectory])
        xs, ys = lonLatToMetric(lon, lat)
        
        if query.trajectory.id == Trajectory:
            plt.plot(xs, ys, linestyle='solid')
        elif Trajectory in ids:
            plt.plot(xs, ys, linestyle='solid', alpha=0.45)
        else:
            plt.plot(xs, ys, linestyle='dashed', alpha= 0.1)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.show()

def plotkNNQuery(trajectories: list[T], query: KNN, rtree):
    # Plots a kNN query
    # Plots the trajectory in the query as a solid line
    # Plots all other trajectories returned by the query as a dashed line
    # Plots any trajectory not returned by the query as a dotted and translucent line

    # Run the query
    q_traj = query.run(rtree)
    
    # Extract trajectory IDs
    ids = []
    for traj in q_traj:
        ids.append(traj.id)

    # Loop over all trajectories in the dataset
    for Trajectory in trajectories:
        lon, lat = trajCoordSplit(trajectories[Trajectory])
        xs, ys = lonLatToMetric(lon, lat)
        
        # Do the differential plotting
        if query.trajectory.id == Trajectory:
            plt.plot(xs, ys, linestyle='solid', alpha=1.0)
        elif Trajectory in ids:
            plt.plot(xs, ys, linestyle='dashed', alpha=1.0)
        else:
            plt.plot(xs, ys, linestyle='dotted', alpha=0.1, figsize=(8, 5))
        
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

    plt.show()

def plotClusterQuery(trajectories: list[T], query: CQ, rtree):
    # Needs refactoring of the abstract class before this works
    pass

def plotF1Scores(f1scores, cr_list, bar_labels):
    fig, ax = plt.subplots()

    input_cr = [str(cr) for cr in cr_list]

    ax.bar(input_cr, f1scores, width=0.3, label=bar_labels)
    ax.set_xlabel("Compression Rate")
    ax.set_ylabel("F1Score")
    ax.set_title("F1Scores by compression rate")

    plt.show()

def plotSimpVsOrig(origTraj, simpTraj):
    # We assume that the input is a tuple of x-coordinates and y-coordinates in both cases.
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(origTraj[0], origTraj[1])
    ax1.set_xlabel("Degrees of latitude")
    ax1.set_ylabel("Degrees of longitude")

    ax2.plot(simpTraj[0], simpTraj[1])
    ax2.set_xlabel("Degrees of latitude")
    ax2.set_ylabel("Degrees of longitude")

    plt.show()

def plotWTA(trajectories: dict, query: RQ, rtree):
    # Plotting function for Winner Takes All

    fig, ax = plt.subplots(figsize=(8,5))

    _, ids = plottingSetup(trajectories, query, rtree)

    # Plot a rectangle thats equivalent to the range specified by the query
    x_range = [query.x1, query.x1, query.x2, query.x2]
    y_range = [query.y1, query.y2, query.y2, query.y1]
    ax.fill(x_range, y_range, alpha=0.3)

    ax.plot(query.centerx, query.centery, marker='o', markersize=5, color='red')

    ax.grid()

    # Loop over all trajectories in the dataset, find the closest point to the query center
    for Trajectory in list(trajectories.keys())[0:math.floor(len(trajectories)/5)]:
        xs, ys = trajCoordSplit(trajectories[Trajectory])

        # Plot the trajectory
        ax.plot(xs, ys, linestyle='solid')

        if not trajectories[Trajectory].id in ids:
            continue        

        min_dist = math.inf
        closest_point = None

        # Find the closest point to the query center
        for i in range(len(xs)):
            dist = np.linalg.norm([(xs[i] - query.centerx), (ys[i] - query.centery)])
            if dist < min_dist:
                min_dist = dist
                closest_point = (xs[i], ys[i])

        # Plot the closest point
        if closest_point:
            ax.plot(closest_point[0], closest_point[1], marker='o', markersize=5, color='blue')
            ax.plot([closest_point[0], query.centerx], [closest_point[1], query.centery], linestyle='dotted')
    
    plt.xlabel("Degrees of Longitude")
    plt.ylabel("Degrees of Latitutude")

    plt.show()

def plotSE(trajectories: list[T], query: RQ, rtree):
    # Plotting function for Winner Takes All

    fig, ax = plt.subplots(figsize=(8,5))

    # Plot a rectangle thats equivalent to the range specified by the query
    x_range = [query.x1, query.x1, query.x2, query.x2]
    y_range = [query.y1, query.y2, query.y2, query.y1]
    ax.fill(x_range, y_range, alpha=0.3)

    # Loop over all trajectories in the dataset, find the closest point to the query center
    for Trajectory in trajectories[0:(len(trajectories)/5)]:
        xs, ys = trajCoordSplit(trajectories[Trajectory])
        #xs, ys = [], []

        #for (x, y) in zip(lon, lat):
        #    temp = lonLatToMetric(x, y)
        #    xs.append(temp[0])
        #    ys.append(temp[1])

        # Plot the trajectory
        ax.plot(xs, ys, linestyle='dotted', alpha=0.3)

        # If node is in the range, plot it
        for node in trajectories[Trajectory].nodes:
            if query.x1 <= node.x <= query.x2 and query.y1 <= node.y <= query.y2:
                ax.plot(node.x, node.y, marker='o', markersize=2, color='blue')
    
    plt.xlabel("Degrees of Longitude")
    plt.xlabel("Degrees of Latitutude")

    plt.show()

def plotGP(trajectories: list[T], query: RQ, rtree):
    # Plotting function for Gradient Points
    # Plotting function for Winner Takes All

    # Plot a rectangle thats equivalent to the range specified by the query
    x_range = [query.x1, query.x1, query.x2, query.x2]
    y_range = [query.y1, query.y2, query.y2, query.y1]

    plt.plot(query.centerx, query.centery, marker='o', markersize=5, color='red')

    for i in range(1, 100):
        xs = [query.centerx + (((xVal - query.centerx) * i)/100) for xVal in x_range]
        ys = [query.centery + (((yVal - query.centery) * i)/100) for yVal in y_range]
        plt.fill(xs, ys, alpha=(1/i), color='blue')

    # Loop over all trajectories in the dataset, find the closest point to the query center
    for Trajectory in trajectories[0:(len(trajectories)/5)]:
        xs, ys = trajCoordSplit(trajectories[Trajectory])
        #xs, ys = [], []

        #for (x, y) in zip(lon, lat):
        #    temp = lonLatToMetric(x, y)
        #    xs.append(temp[0])
        #    ys.append(temp[1])

        # Plot the trajectory
        plt.plot(xs, ys, linestyle='solid', alpha=1)
    
    plt.show()


def plotGP2(trajectories: list, query: RQ, rtree):
    # Match your desired figure shape (landscape)
    fig, ax = plt.subplots(figsize=(8,5))

    # Plot trajectories first
    all_xs, all_ys = [], []
    for Trajectory in trajectories[0:(len(trajectories)/5)]:
        xs, ys = trajCoordSplit(trajectories[Trajectory])
        all_xs.extend(xs)
        all_ys.extend(ys)
        ax.plot(xs, ys, linestyle='solid', alpha=1)

    # Plot query center
    ax.plot(query.centerx, query.centery, marker='o', markersize=5, color='red')

    # Save limits for restoring after gradient
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Generate circular gradient
    resolution = 500
    x = np.linspace(query.x1, query.x2, resolution)
    y = np.linspace(query.y1, query.y2, resolution)
    xx, yy = np.meshgrid(x, y)

    dx = xx - query.centerx
    dy = yy - query.centery
    radius = np.sqrt(dx**2 + dy**2)

    max_radius = np.sqrt(((query.x2 - query.x1)/2)**2 + ((query.y2 - query.y1)/2)**2)
    norm_radius = np.clip(radius / max_radius, 0, 1)
    gradient = 1 - norm_radius

    # Draw the gradient without distorting axis
    ax.imshow(
        gradient,
        extent=(query.x1, query.x2, query.y1, query.y2),
        origin='lower',
        cmap='Blues',
        alpha=1,
        interpolation='bilinear',
        zorder=0
    )

    # Restore limits to match trajectory view
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Let matplotlib handle aspect ratio automatically
    # Don't use ax.set_aspect('equal')

    # Optionally match tick style if needed
    ax.tick_params(labelsize=10)

    plt.xlabel("Degrees of Longitude")
    plt.xlabel("Degrees of Latitutude")

    #plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    plotF1Scores([1, 0.9, 0.8], [0.9, 0.85, 0.8])