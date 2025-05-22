import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
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

def parseDistributeCombination(filename: str):
    # Parses the filename of the 'pkl' file to extract the exact combination of distribute methods used
    # Returns a list of names of the distribute methods used
    
    # Example filename: "scores_knn1_range1_simc.pkl"
    # Example output: ["RangeWTA", "SimilarClosest"]
    # Note that kNN is not included in the output, since there is only one variant of kNN distribute method
    # and it is always used

    # Split the filename by underscores
    parts = re.split(r'[_.]', filename)
    #print(parts)
    rangeType, similarType = None, None

    for part in parts:
        if part == "range1":
            rangeType = "RangeWTA"
        if part == "range2":
            rangeType = "RangeOneEven"
        if part == "range3":
            rangeType = "RangeFracEven"
        if part == "range4":
            rangeType = "RangeGradient"
        if part == "sima":
            similarType = "SimilarAll"
        if part == "simc":
            similarType = "SimilarClosest"
        if part == "simc+f":
            similarType = "SimilarClosestAndFurthest"
        if part == "simm":
            similarType = "SimilarRecedes"
    
    return rangeType, similarType

def plotF1Scores(f1scores, cr_list, width=0.15, bar_labels=['averageF1Score', 'rangeF1', 'similarityF1', 'kNNF1'],
                 color_list=["red", "blue", "orange", "green"], distribute_variants=None, plotname="" , show=False):
    fig, ax = plt.subplots(figsize=(16,10))

    br0 = np.arange(len(cr_list)) 
    br1 = [x - 1.5 * width for x in br0]
    br2 = [x - 0.5 * width for x in br0]
    br3 = [x + 0.5 * width for x in br0] 
    br4 = [x + 1.5 * width for x in br0]
    
    ax.bar(br1, f1scores[0], width=width, label=bar_labels[0], color=color_list[0])
    ax.bar(br2, f1scores[1], width=width, label=bar_labels[1], color=color_list[1])
    ax.bar(br3, f1scores[2], width=width, label=bar_labels[2], color=color_list[2])
    ax.bar(br4, f1scores[3], width=width, label=bar_labels[3], color=color_list[3])

    ax.set_xlabel("Compression Rate", fontsize=14)
    ax.set_ylabel("F1Score", fontsize=14)

    ax.set_xticks([x for x in range(len(cr_list))], [str(x) for x in cr_list])
    
    ax.set_title(f"F1Scores by compression rate using {distribute_variants[0]} and {distribute_variants[1]}", fontweight='bold', fontsize=14)

    ax.legend()

    if show:
        plt.show()
    else:
        tempPath = os.path.dirname(__file__)
        path = os.path.join(tempPath, "../Plots")
        plt.savefig(f"{path}\Bar Plots\{plotname}.png", bbox_inches='tight')
        plt.close()

def plotResultHeatmap(data, xlabels, ylabels, f1score_type="F1 Score", show=False):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(data, cmap='Blues', label='F1 Score')

    ax.set_title(f"Heatmap of {f1score_type} in relation to \n compression rate and distribution method", fontweight='bold', fontsize=14)

    ax.set_xlabel("Compression Rate", fontsize=14)
    ax.set_ylabel("Distribution Methods", fontsize=14)

    ax.set_xticks(range(len(data[0])), xlabels)
    ax.set_yticks(range(len(data)), ylabels)
    
    temp = f1score_type.split(" ")
    saveName = '_'.join(temp)
    
    if show:
        plt.show()
    else:
        tempPath = os.path.dirname(__file__)
        path = os.path.join(tempPath, "../Plots")
        plt.savefig(f"{path}\Heatmaps\heatmap_{saveName}.png", bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    df = pd.read_csv('scores_summary - scores_summary.csv')

    average_heatmap_matrix = []
    range_heatmap_matrix = []
    similarity_heatmap_matrix = []
    knn_heatmap_matrix = []

    i = 0
    x_acc = []
    y_acc = [[], [], [], []]
    names = []

    #print(df.head(1)['file'][0])
    #print(parseDistributeCombination(df.head(1)['file'][0]))
    

    for elem in df.iterrows():
        i += 1
        # print(elem[1][0], elem[1][1], elem[1][2], elem[1][3], elem[1][4])

        x_acc.append(elem[1][2])
        for j in range(4):
            y_acc[j].append(elem[1][j+3])

        if i == 5:
            #print(x_acc)
            #print(y_acc)
            #print(elem[1]['file'])
            names.append(elem[1][1])
            
            average_heatmap_matrix.append(y_acc[0])
            range_heatmap_matrix.append(y_acc[1])
            similarity_heatmap_matrix.append(y_acc[2])
            knn_heatmap_matrix.append(y_acc[3])

            plotname = f"{parseDistributeCombination(elem[1][1])[0]}_and_{parseDistributeCombination(elem[1][1])[1]}_plot"

            plotF1Scores(y_acc, x_acc, distribute_variants=parseDistributeCombination(elem[1]['file']), plotname=plotname, show=False)

            i = 0
            x_acc = []
            y_acc = [[], [], [], []]
            continue

    parsed_names = [f"{parseDistributeCombination(name)[0]}\n{parseDistributeCombination(name)[1]}" for name in names]

    plotResultHeatmap(average_heatmap_matrix, xlabels=[0.8, 0.9, 0.95, 0.975, 0.99], ylabels=parsed_names, f1score_type="average F1 score", show=False)
    plotResultHeatmap(range_heatmap_matrix, xlabels=[0.8, 0.9, 0.95, 0.975, 0.99], ylabels=parsed_names, f1score_type="range F1 score", show=False)
    plotResultHeatmap(similarity_heatmap_matrix, xlabels=[0.8, 0.9, 0.95, 0.975, 0.99], ylabels=parsed_names, f1score_type="similarity F1 score", show=False)
    plotResultHeatmap(knn_heatmap_matrix, xlabels=[0.8, 0.9, 0.95, 0.975, 0.99], ylabels=parsed_names, f1score_type="kNN F1 score", show=False)