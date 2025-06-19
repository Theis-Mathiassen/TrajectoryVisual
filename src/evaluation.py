#import sys
#sys.path.append("src/")
# from src.Node import Node
# from src.Trajectory import Trajectory
from src.clusterQuery import ClusterQuery
from src.knnQuery import KnnQuery
import numpy as np
import numpy.ma as ma
from src.Query import Query
from src.QueryWrapper import QueryWrapper
from itertools import combinations
from tqdm import tqdm
import pickle
import os

CACHE_FILE = "cached_rtree_query_eval_results.pkl"

def load_from_cache(key):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
            return cache.get(key)
    else:
        return None
    
def save_to_cache(key, data):
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
    cache[key] = data
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def runAndGenerateSet(rtree, query, use_cache = False, differentTrajs = None):
    if use_cache:
        cache_key = repr((rtree, query))
        cached_data = load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
    
    # Adjust query to be able to use origin trajectories
    if differentTrajs is not None:
        oldTrajs = query.trajectories
        query.trajectories = differentTrajs
    result = query.run(rtree, query.trajectories)
    
    if differentTrajs is not None:
        query.trajectories = oldTrajs

    if isinstance(query, KnnQuery):
        set_result = set([item.id for item in result])
    else:
        set_result = set([trajectory_id for trajectory_id, _ in result])
    
    if use_cache:
        save_to_cache(cache_key, set_result)

    return set_result


# This code allows testing of simplified trajectories
def getClusterSet(rtree, query, use_cache = False, differentTrajs = None):
    if use_cache:
        cache_key = repr((rtree, query))
        cached_data = load_from_cache(cache_key)
        
        if cached_data is not None:
            return cached_data
        
    # Adjust query to be able to use origin trajectories
    if differentTrajs is not None:
        oldTrajs = query.trajectories
        query.trajectories = differentTrajs
        
    clusters = query.run(rtree)
    
    if differentTrajs is not None:
        query.trajectories = oldTrajs

    # for cluster in clusters:
    #     cluster = [trajectory.id for trajectory in cluster]

    result = set()
    for cluster in clusters:
        result.update(tuple(sorted(pair)) for pair in combinations(cluster, 2)) # Gets all combinations of 2 trajectories, removing duplicates
    if use_cache:
        save_to_cache(cache_key, result)
    
    return result
    
def getIntersection(trajectoryList1, trajectoryList2):
    return list(set(trajectoryList1[0]) & set(trajectoryList2[0]))
    return [trajectory for trajectory in trajectoryList1.values() if trajectory.id in [trajectory.id for trajectory in trajectoryList2.values()]]

def getF1Score(query : Query, rtree_original, rtree_simplified, trajectories_original):

    # Cluster queries must be handled differently. Alternatively handle them in a different function
    if isinstance(query, ClusterQuery):
        #print('ClusterQuery is not implemented yet.')
        setOriginal_result = getClusterSet(rtree_original, query, use_cache=True, differentTrajs=trajectories_original)
        setSimplified_result = getClusterSet(rtree_simplified, query)

    else: # For all other queries
        
        setOriginal_result = runAndGenerateSet(rtree_original, query, use_cache=True, differentTrajs=trajectories_original)
        setSimplified_result = runAndGenerateSet(rtree_simplified, query)


    intersection = setOriginal_result & setSimplified_result

    if len(setOriginal_result) == 0 and len(setSimplified_result) == 0:
        return 1
    elif (len(setOriginal_result) == 0 or len(setSimplified_result) == 0 or len(intersection) == 0):
        return 0
    
    precision = len(intersection) / len(setSimplified_result)
    recall = len(intersection) / len(setOriginal_result)

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def getAverageF1ScoreAll(queryWrapper : QueryWrapper, rtree_original, rtree_simplified, trajectories_original):
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

        for query in tqdm(listOfQueries, desc=f"Running {queryTypeString} queries"):
            f1_score += getF1Score(query, rtree_original, rtree_simplified, trajectories_original)

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
    sim_traj = [[node.x, node.y, node.t] for node in sim_traj.nodes.compressed()]
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
    # Maybe use the masked arrays instead
    ori_traj = [[node.x, node.y, node.t] for node in ori_traj.nodes]
    sim_traj = [[node.x, node.y, node.t] for node in sim_traj.nodes.compressed()]
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

    for key in original_trajectory_list.keys():
        avg_SED += sed_error(original_trajectory_list.get(key), simplified_trajectory_list.get(key))[1]
        avg_PED += ped_error(original_trajectory_list.get(key), simplified_trajectory_list.get(key))[1]

    avg_SED /= length
    avg_PED /= length

    return avg_SED, avg_PED 
