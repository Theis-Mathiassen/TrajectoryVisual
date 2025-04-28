from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
from src.Util import ParamUtil
from src.QueryWrapper import QueryWrapper
from src.scoringQueries import giveQueryScorings
from src.load import build_Rtree, load_Tdrive, loadRtree, load_Tdrive_Rtree, get_Tdrive
from src.dropNodes import dropNodes
from src.clusterQuery import ClusterQuery

import os
import sys
import copy
import pickle
import math
from tqdm import tqdm
import pandas as pd 
import os
import logging
import traceback # traceback for information on python stack traces

sys.path.append("src/")

CSVNAME = 'first_10000_train'
DATABASENAME = 'original_Taxi'
SIMPLIFIEDDATABASENAME = 'simplified_Taxi'
LOG_FILENAME = 'script_error_log.log' # Define a log file name
PICKLE_HITS = ['RangeQueryHits.pkl']


logging.basicConfig(
    level=logging.ERROR, # Log only ERROR level messages and above
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILENAME, # Log to this file
    filemode='a' # Append mode, use 'w' to overwrite each time
)
logger = logging.getLogger(__name__)

#### main
def main(config):

    origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)

    """     ORIGTrajectories = {
        tid : copy.deepcopy(traj)
        for tid, traj, in tqdm(origTrajectories.items(), desc = "Copying trajectories")
    } """

    ## Setup data collection environment, that is evaluation after each epoch

    # ---- Set number of queries to be created ----
    if config["QueriesPerTrajectory"] != None : config["numberOfEachQuery"] = math.floor(config["QueriesPerTrajectory"] * len(origTrajectories.values()))


    # ---- Create training queries -----
    origRtreeQueriesTraining : QueryWrapper = QueryWrapper(config["numberOfEachQuery"], random=False, trajectories=origTrajectories)
    origRtreeParamsTraining : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours
    

    # origRtreeQueriesTraining.createRangeQueries(origRtree, origRtreeParamsTraining)
    # origRtreeQueriesTraining.createSimilarityQueries(origRtree, origRtreeParamsTraining)
    origRtreeQueriesTraining.createKNNQueries(origRtree, origRtreeParamsTraining)
    # origRtreeQueriesTraining.createClusterQueries(origRtree, origRtreeParamsTraining)



    
    queryResults = []
    
    
    for Query in tqdm(origRtreeQueriesTraining.getQueries(), desc="Running queries"):#[queryWrapper.RangeQueries + queryWrapper.KNNQueries + queryWrapper.SimilarityQueries + queryWrapper.ClusterQueries]:
        # Get result of query
        result = Query.run(origRtree)
        # Distribute points
        queryResults.append((Query, result))
    #print("Done!")

    dictQueryResults = {}

    for q, r in queryResults:
        if str(q) not in dictQueryResults:
            dictQueryResults[str(q)] = []
        dictQueryResults[str(q)].append((q,r))
    
    for queryType in dictQueryResults.keys():    
        try:
            with open(os.path.join(os.getcwd(), str(queryType) + 'Hits.pkl'), 'wb') as file:
                pickle.dump(dictQueryResults[queryType], file)
                file.close()
        except Exception as e:
            print(f"err saving results: {e}")
            logger.error("problems when saving results to pickle file.")
            logger.error(traceback.format_exc()) 
    pass




if __name__ == "__main__":
    config = {}
    config["epochs"] = 100                  # Number of epochs to simplify the trajectory database
    config["compression_rate"] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]      # Compression rate of the trajectory database
    config["DB_size"] = 100                 # Amount of trajectories to load (Potentially irrelevant)
    config["verbose"] = True                # Print progress
    config["trainTestSplit"] = 0.8          # Train/test split
    config["numberOfEachQuery"] = 5     # Number of queries used to simplify database    
    config["QueriesPerTrajectory"] = 1   # Number of queries per trajectory, in percentage. Overrides numberOfEachQuery if not none

    print("Script starting...") 
    try:
        main(config)
        print("Script finished successfully.") 

    except Exception as e:
        print(f"\n--- SCRIPT CRASHED ---")
        print(f"!!! error occurred: {e}")
        print(f"See {LOG_FILENAME} for detailed err traces.")

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
            