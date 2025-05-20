#from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
from pathlib import Path
from src.Util import ParamUtil
from src.QueryWrapper import QueryWrapper
from src.scoringQueries import giveQueryScorings
from src.load import build_Rtree, get_geolife, load_Tdrive, loadRtree, load_Tdrive_Rtree, get_Tdrive
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
import logging
import traceback # traceback for information on python stack traces

sys.path.append("src/")
output_dir = os.environ.get('JOB_OUTPUT_DIR', os.getcwd());
Path(output_dir).mkdir(parents=True, exist_ok=True)

CSVNAME = 'first_10000_train'
DATABASENAME = 'geolife'
SIMPLIFIEDDATABASENAME = 'simplified_Taxi'

USE_GAUSSIAN = True

#### main
def main(config):
    logger.info('Using guassian: '+ str(USE_GAUSSIAN))

    logger.info('Starting load for: ' + DATABASENAME)
    origRtree, origTrajectories = get_geolife(filename=DATABASENAME)
    logger.info('Completed loading: ' + DATABASENAME)
    """     
    logger.info('Copying trajectories.')
    ORIGTrajectories = {
        tid : copy.deepcopy(traj)
        for tid, traj, in tqdm(origTrajectories.items(), desc = "Copying trajectories")
    } """

    ## Setup data collection environment, that is evaluation after each epoch

    # ---- Set number of queries to be created ----
    if config["QueriesPerTrajectory"] != None : config["numberOfEachQuery"] = math.floor(config["QueriesPerTrajectory"] * len(origTrajectories.values()))


    avgCoordinatesValue = None
    stdCoordinatesValue = None
    if USE_GAUSSIAN:
        avgCoordinatesValue = getAverageNodeCoordinates(origTrajectories)



    # ---- Create training queries -----
    logger.info('Creating training queries.')

    origRtreeQueriesTraining : QueryWrapper = QueryWrapper(config["numberOfEachQuery"], random=False, trajectories=origTrajectories, useGaussian=USE_GAUSSIAN, avgCoordinateValues=avgCoordinatesValue, rtree=origRtree, sigma=500)
    origRtreeParamsTraining : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours
    
    # Create the specified query type based on command line argument
    query_type = config["query_type"].lower()
    print(f"Creating {query_type} queries...")
    
    if query_type == "range":
        logger.info('Creating range queries.')
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
            with open(os.path.join(output_dir, gaussianExtra + str(queryType) + 'Hits_geolife.pkl'), 'wb') as file:
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
    config["QueriesPerTrajectory"] = 1   # Number of queries per trajectory, in percentage. Overrides numberOfEachQuery if not none
    config["query_type"] = args.query_type  # Query type from command line args

    try:
        logger.info("Script starting...")
        main(config)
        logger.info("Script finished successfully.")

    except Exception as e:
        print(f"\n--- SCRIPT CRASHED ---")
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
            


