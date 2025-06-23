#from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
from pathlib import Path
from src.Util import ParamUtil
from src.QueryWrapper import QueryWrapper
from src.scoringQueries import giveQueryScorings
from src.load import build_Rtree, load_Tdrive, loadRtree, load_Tdrive_Rtree, get_Tdrive
from src.dropNodes import dropNodes
from src.clusterQuery import ClusterQuery
#import logging
from src.log import logger, ERROR_LOG_FILENAME
import os
import sys
import copy
import pickle
import math
import argparse
from tqdm import tqdm
import pandas as pd 
import numpy as np
import logging
import traceback # traceback for information on python stack traces

sys.path.append("src/")
output_dir = os.environ.get('JOB_OUTPUT_DIR', os.getcwd());
Path(output_dir).mkdir(parents=True, exist_ok=True)

CSVNAME = 'first_10000_train'
DATABASENAME = 'original_Taxi'
SIMPLIFIEDDATABASENAME = 'simplified_Taxi'
LOG_FILENAME = 'script_error_log.log' # Define a log file name
PICKLE_HITS = ['RangeQueryHits.pkl']

USE_GAUSSIAN = False
USE_DATADISTRIBUTION = True

#### main
def main(config):

    #load_Tdrive(CSVNAME + '.csv', CSVNAME + '_trimmed.csv')

    #origRtree, origTrajectories = build_Rtree(CSVNAME + '_trimmed.csv', filename=DATABASENAME)

         
    logger.info('Starting get_Tdrive.')
    origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)
    logger.info('Completed get_Tdrive.')
    #logger.info('Copying trajectories.')
    """ORIGTrajectories = {
        tid : copy.deepcopy(traj)
        for tid, traj, in tqdm(origTrajectories.items(), desc = "Copying trajectories")
    } """

    ## Setup data collection environment, that is evaluation after each epoch

    # ---- Set number of queries to be created ----
    if config["QueriesPerTrajectory"] is not None:
        config["numberOfEachQuery"] = math.floor(config["QueriesPerTrajectory"] * len(origTrajectories.values()))


    avgCoordinatesValue = None
    stdCoordinatesValue = None
    if USE_GAUSSIAN:
        avgCoordinatesValue = getAverageNodeCoordinates(origTrajectories)
        stdCoordinatesValue = getStandardDerivationNodeCoordinates(origTrajectories, avgCoordinatesValue) * 0.25
        print(avgCoordinatesValue)
        print(stdCoordinatesValue)

    dataTrajGrid, dataNodeGrid = getDataDistribution(origTrajectories, 2000, 10800)
    listTrajCellTup = list(dataTrajGrid.keys())
    listTrajCell = np.array(list(dataTrajGrid.values()))
    listTrajCellCnt = np.array([len(cell) for cell in list(dataTrajGrid.values())])
    totalTrajCellCnt = np.sum(listTrajCellCnt)
    #listTrajCellProbDist = list
    #sampleT    rajCellDist = 
    
    keys = np.random.choice(len(listTrajCellTup), config["numberOfEachQuery"], p=(listTrajCellCnt/totalTrajCellCnt))
    listSampleCellKeys = [listTrajCellTup[key] for key in keys]
    


    # ---- Create training queries -----
    logger.info('Creating training queries.')

    origRtreeQueriesTraining : QueryWrapper = QueryWrapper(config["numberOfEachQuery"], random=False, trajectories=origTrajectories, useGaussian=USE_GAUSSIAN, avgCoordinateValues=avgCoordinatesValue, rtree=origRtree, sigma=stdCoordinatesValue)
    origRtreeParamsTraining : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours
    
    # Create the specified query type based on command line argument
    query_type = config["query_type"].lower()
    print(f"Creating {query_type} queries...")
    
    if query_type == "range":
        logger.info('Creating range queries.')
        if USE_DATADISTRIBUTION:
            origRtreeQueriesTraining.createRangeQueries(origRtree, origRtreeParamsTraining, cellDist=listSampleCellKeys)
        else:
            origRtreeQueriesTraining.createRangeQueries(origRtree, origRtreeParamsTraining)
    elif query_type == "similarity":
        logger.info('Creating similarity queries.')
        origRtreeQueriesTraining.createSimilarityQueries(origRtree, origRtreeParamsTraining)
    elif query_type == "knn":
        logger.info('Creating KNN queries.')
        origRtreeQueriesTraining.createKNNQueries(origRtree, origRtreeParamsTraining)
    elif query_type == "cluster":
        logger.info('Creating cluster queries.')
        origRtreeQueriesTraining.createClusterQueries(origRtree, origRtreeParamsTraining)
    else:
        print(f"Unknown query type: {query_type}")
        print("Available query types: range, similarity, knn, cluster")
        sys.exit(1)

    queryResults = []
    
    
    for Query in tqdm(origRtreeQueriesTraining.getQueries(), desc="Running queries"):#[queryWrapper.RangeQueries + queryWrapper.KNNQueries + queryWrapper.SimilarityQueries + queryWrapper.ClusterQueries]:
        # Get result of query
        logger.info('Running query %s', type(Query))
        result = Query.run(origRtree, origTrajectories)
        # Distribute points
        queryResults.append((Query, result))
    #print("Done!")

    dictQueryResults = {}

    for q, r in queryResults:
        if str(q) not in dictQueryResults:
            dictQueryResults[str(q)] = []
        dictQueryResults[str(q)].append((q ,r))
    
    logger.info('Saving results to pickle files.')

    gaussianExtra = ""
    if USE_GAUSSIAN:
        gaussianExtra = "Gaussian"


    for queryType in dictQueryResults.keys():    
        try:
            with open(os.path.join(output_dir, gaussianExtra + str(queryType) + 'Hits.pkl'), 'wb') as file:
                pickle.dump(dictQueryResults[queryType], file)
                file.close()
        except Exception as e:
            print(f"err saving results: {e}")
            logger.error("problems when saving results to pickle file.")
            logger.error(traceback.format_exc()) 
    pass


def getAverageNodeCoordinates(trajectories):
    logger.info("Using Gaussian distribution for creating queries instead of uniform distribution.")

    # Find average location
    totalNodes = 0
    avgx = 0
    avgy = 0
    avgt = 0
    for traj in tqdm(trajectories.values(), desc="Finding average location of nodes"):
        nodes = traj.nodes.compressed()
        totalNodes += len(nodes)
        for node in nodes:
            avgx += node.x
            avgy += node.y
            avgt += node.t

    avgx /= totalNodes
    avgy /= totalNodes
    avgt /= totalNodes

    avgCoordinateValues = [avgx, avgy, avgt]

    return avgCoordinateValues


def getStandardDerivationNodeCoordinates(trajectories, averages):
    logger.info("Using Gaussian distribution for creating queries instead of uniform distribution.")

    # Find average location
    totalNodes = 0
    stdx = 0
    stdy = 0
    stdt = 0
    for traj in tqdm(trajectories.values(), desc="Finding squared difference of nodes with mean"):
        nodes = traj.nodes.compressed()
        nodesx = np.array([node.x for node in nodes])
        nodesy = np.array([node.y for node in nodes])
        nodest = np.array([node.t for node in nodes])
        totalNodes += len(nodes)
        stdx += np.sum((nodesx - averages[0]) ** 2)
        stdy += np.sum((nodesy - averages[1]) ** 2)
        stdt += np.sum((nodest - averages[2]) ** 2)
        
        """for node in nodes:
            avgx += node.x
            avgy += node.y
            avgt += node.t"""

    stdx /= totalNodes
    stdy /= totalNodes
    stdt /= totalNodes


    stdCoordinateValues = np.sqrt([stdx, stdy, stdt])

    return stdCoordinateValues

def getDataDistribution(trajectories, spatialWindow, temporalWindow):
    dataTrajGrid = {}
    dataNodeGrid = {}
    
    logger.info('Creating data distribution')
    for trajectory in tqdm(trajectories.values()):
        for node in trajectory.nodes:
            tup = (node.x // spatialWindow, node.y // spatialWindow, node.t // temporalWindow)
            if tup not in dataTrajGrid:
                dataTrajGrid[tup] = set()
            if tup not in dataNodeGrid:
                dataNodeGrid[tup] = set()
            dataNodeGrid[tup].add((trajectory.id, node.id))
            dataTrajGrid[tup].add(trajectory.id)
    
    return dataTrajGrid, dataNodeGrid

if __name__ == "__main__":
    logger.info("---------------------------    Train.py Main Start    ---------------------------")
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run trajectory queries with specified query type')
    parser.add_argument('query_type', type=str, help='Query type to run (range, similarity, knn, or cluster)')
    args = parser.parse_args()
    
    config = {}
    config["compression_rate"] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]      # Compression rate of the trajectory database
    config["DB_size"] = 100                 # Amount of trajectories to load (Potentially irrelevant)
    config["verbose"] = True                # Print progress
    config["trainTestSplit"] = 0.8          # Train/test split
    config["numberOfEachQuery"] = 100     # Number of queries used to simplify database    
    config["QueriesPerTrajectory"] = None   # Number of queries per trajectory, in percentage. Overrides numberOfEachQuery if not none
    config["query_type"] = args.query_type  # Query type from command line args

    try:
        logger.info("Script starting...")
        main(config)
        logger.info("Script finished successfully.")

    except Exception as e:
        print("\n--- SCRIPT CRASHED ---")
        print(f"!!! error occurred: {e}")
        print(f"See {ERROR_LOG_FILENAME} for detailed err traces.")

        # Log the exception information to the file
        logger.error(f"Script crashed with the following error: {e}")
        logger.error("Trace:\n%s", traceback.format_exc()) # Log the full traceback

    finally:
        print("\n execution finished (i.e. it either completed or crashed).")

    
    # Cleanup temporary cache files
    filesToClear = ["cached_rtree_query_eval_results.pkl"]

    for fileString in filesToClear:
        if os.path.exists(fileString):
            os.remove(fileString)
            


