import sys
sys.path.append("src/")
from Node import Node
from Trajectory import Trajectory
from clusterQuery import ClusterQuery
import numpy as np
from Query import Query
from QueryWrapper import QueryWrapper
from tqdm import tqdm

# This code allows testing of simplified trajectories

def getIntersection(trajectoryList1, trajectoryList2):
    return [trajectory for trajectory in trajectoryList1 if trajectory.id in [trajectory.id for trajectory in trajectoryList2]]

def getF1Score(Query : Query, rtree_original, rtree_simplified):

    # Cluster queries must be handled differently. Alternatively handle them in a different function
    if Query is ClusterQuery:
        print('ClusterQuery is not implemented yet.')
        return 0

    original_result = Query.run(rtree_original)
    simplified_result = Query.run(rtree_simplified)
    
    if (len(original_result) == 0 or len(simplified_result) == 0):
        return 0

    intersection = getIntersection(original_result, simplified_result)

    precision = len(intersection) / len(simplified_result)
    recall = len(intersection) / len(original_result)

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def getAverageF1ScoreAll(queryWrapper : QueryWrapper, rtree_original, rtree_simplified):
    """
    Runs queries and returns average F1Scores.

    :returns averageF1Score: Average f1 score for all queries
    :returns rangeF1Score: Average f1 score for range queries
    :returns similarityF1Score: Average f1 score for similarity queries
    :returns KNNF1Score: Average f1 score for KNN queries
    :returns clusterF1Score: Average f1 score for clustering queries
    """
    # Gets average f1 score

    rangeQueries = queryWrapper.RangeQueries
    similarityQueries = queryWrapper.SimilarityQueries
    KNNQueries = queryWrapper.KNNQueries
    clusterQueries = queryWrapper.ClusterQueries

    totalLength = 0
    totalF1Score = 0

    def getQueryF1Score(listOfQueries, queryTypeString):
        nonlocal totalLength, totalF1Score # allow modification of the outer variables in enclosing scope
        length = len(listOfQueries)
        f1_score = 0

        if length == 0:
            return 0

        print(f"Running {queryTypeString} queries..")
        for query in tqdm(listOfQueries):
            f1_score += getF1Score(query, rtree_original, rtree_simplified)

        totalLength += length   # Increment
        totalF1Score += f1_score

        return f1_score / length

    rangeF1Score = getQueryF1Score(rangeQueries, "range")
    similarityF1Score = getQueryF1Score(similarityQueries, "similarity")
    KNNF1Score = getQueryF1Score(KNNQueries, "KNN")
    clusterF1Score = getQueryF1Score(clusterQueries, "cluster")

    averageF1Score = totalF1Score / totalLength if totalLength > 0 else 0

    return averageF1Score, rangeF1Score, similarityF1Score, KNNF1Score, clusterF1Score



# Synchronized euclidean distance
# euclidean distance between the actual location p_i, and its syncrhonized node on the anchor segment
# where synchronized refers to estimated point on the anchor segment at time p_i
def sed_op(segment):
    if len(segment) <= 2:
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        for i in range(1, len(segment) - 1):
            syn_time = segment[i][2]
            time_ratio = 1 if (pe[2] - ps[2]) == 0 else (syn_time - ps[2]) / (pe[2] - ps[2])
            syn_x = ps[0] + (pe[0] - ps[0]) * time_ratio
            syn_y = ps[1] + (pe[1] - ps[1]) * time_ratio
            e = max(e, np.linalg.norm(np.array([segment[i][0], segment[i][1]]) - np.array([syn_x, syn_y])))
        # print('segment error', e)
        return e

# segment error for a single trajectory
# This code finds the points between retained points. And then finds the max segment error for them
# See https://github.com/yumengs-exp/MLSimp/blob/main/Utils/query_utils_val.py :90
def sed_error(ori_traj, sim_traj):
    # Convert code first
    ori_traj = [[node.x, node.y, node.t] for node in ori_traj.nodes]
    sim_traj = [[node.x, node.y, node.t] for node in sim_traj.nodes]
    #Original code

    # ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            # print(start, c)
            error = max(error, sed_op(ori_traj[start: c + 1]))
            start = c
    return t_map, error

# Perpendicular euclidean distance
# Shortest euclidean distance from node to anchor segment
def ped_op(segment):
    if len(segment) <= 2:
        return 0.0
    else:
        ps = segment[0]
        pe = segment[-1]
        e = 0.0

        A = pe[1] - ps[1]
        B = ps[0] - pe[0]
        C = pe[0] * ps[1] - ps[0] * pe[1]

        if A == 0 and B == 0:
            e = max(e, 0.0)
        else:
            for i in range(1, len(segment) - 1):
                pm = segment[i]
                e = max(e, abs((A * pm[0] + B * pm[1] + C) / np.sqrt(A * A + B * B)))
        return e


def ped_error(ori_traj, sim_traj):
    # Convert code first
    ori_traj = [[node.x, node.y, node.t] for node in ori_traj.nodes]
    sim_traj = [[node.x, node.y, node.t] for node in sim_traj.nodes]
    #Original code
    # ori_traj, sim_traj = [[x,y,t],...,[x,y,t]]
    # 1-keep and 0-drop
    dict_traj = {}
    t_map = [0 for i in range(len(ori_traj))]
    for c, value in enumerate(ori_traj):
        dict_traj[tuple(value)] = c
    for value in sim_traj:
        t_map[dict_traj[tuple(value)]] = 1
    error = 0.0
    start = 0
    for c, value in enumerate(t_map):
        if value == 1:
            # print(start, c)
            error = max(error, ped_op(ori_traj[start: c + 1]))
            start = c
    return t_map, error


def GetSimplificationError(original_trajectory_list, simplified_trajectory_list):

    length = len(original_trajectory_list)

    if length != len(simplified_trajectory_list):
        print("The number of original and simplified trajectories are not the same.")
        return -1, -1

    # get average SED and PED error
    avg_SED = 0
    avg_PED = 0

    for i in range(length):
        avg_SED += sed_error(original_trajectory_list[i], simplified_trajectory_list[i])[1]
        avg_PED += ped_error(original_trajectory_list[i], simplified_trajectory_list[i])[1]

    avg_SED /= length
    avg_PED /= length

    return avg_SED, avg_PED 
