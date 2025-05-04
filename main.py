from src.gridSearch import createConfigs
from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
from src.Util import ParamUtil
from src.QueryWrapper import QueryWrapper
from src.scoringQueries import giveQueryScorings
from src.load import build_Rtree, load_Tdrive, loadRtree, load_Tdrive_Rtree, get_Tdrive
from src.dropNodes import dropNodes

import os
import sys
import copy
import pickle
import math
from tqdm import tqdm
import pandas as pd
import os
#import logging
from src.log import logger, ERROR_LOG_FILENAME
import traceback # traceback for information on python stack traces

sys.path.append("src/")

CSVNAME = 'first_10000_train'
DATABASENAME = 'original_Taxi'
SIMPLIFIEDDATABASENAME = 'simplified_Taxi'
LOG_FILENAME = 'script_error_log.log' # Define a log file name


logging.basicConfig(
    level=logging.ERROR, # Log only ERROR level messages and above
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILENAME, # Log to this file
    filemode='a' # Append mode, use 'w' to overwrite each time
)
logger = logging.getLogger(__name__)

# Prepare RTrees for training and testing
def prepareRtree(config, origRtree, origTrajectories):
    # ---- Create training queries -----
    origRtreeQueriesTraining : QueryWrapper = QueryWrapper(math.ceil(config.numberOfEachQuery * config.trainTestSplit))
    origRtreeParamsTraining : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours


    origRtreeQueriesTraining.createRangeQueries(origRtree, origRtreeParamsTraining)
    origRtreeQueriesTraining.createSimilarityQueries(origRtree, origRtreeParamsTraining)
    origRtreeQueriesTraining.createKNNQueries(origRtree, origRtreeParamsTraining)
    # origRtreeQueriesTraining.createClusterQueries(origRtree, origRtreeParamsTraining)

    # ---- Create evaluation queries -----
    origRtreeQueriesEvaluation : QueryWrapper = QueryWrapper(math.floor(config.numberOfEachQuery - config.numberOfEachQuery * config.trainTestSplit))
    origRtreeParamsEvaluation : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours

    origRtreeQueriesEvaluation.createRangeQueries(origRtree, origRtreeParamsEvaluation)
    origRtreeQueriesEvaluation.createSimilarityQueries(origRtree, origRtreeParamsEvaluation)
    origRtreeQueriesEvaluation.createKNNQueries(origRtree, origRtreeParamsEvaluation)
    # origRtreeQueriesEvaluation.createClusterQueries(origRtree, origRtreeParamsEvaluation)
    return origRtreeQueriesTraining, origRtreeQueriesEvaluation



#### main
def main(config):
    ## Load Dataset
    #load_Tdrive(CSVNAME + '.csv', CSVNAME + '_trimmed.csv')


    #origRtree, origTrajectories = build_Rtree(CSVNAME + '_trimmed.csv', filename=DATABASENAME)
    logger.info('Starting get_Tdrive.')
    origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)
    logger.info('Completed get_Tdrive.')
    # simpRtree, simpTrajectories = build_Rtree("first_10000_train_trimmed.csv", filename="simplified_Taxi")
    ## Setup reinforcement learning algorithms (t2vec, etc.)

    #ORIGTrajectories = copy.deepcopy(origTrajectories)
    logger.info('Copying trajectories.')
    ORIGTrajectories = {
        tid : copy.deepcopy(traj)
        for tid, traj, in tqdm(origTrajectories.items(), desc = "Copying trajectories")
    }

    ## Setup data collection environment, that is evaluation after each epoch

    # ---- Set number of queries to be created ----

    if config.QueriesPerTrajectory != None:
        config.numberOfEachQuery = math.floor(config.QueriesPerTrajectory * len(origTrajectories.values()))

    print(f"\n\nNumber of queries to be created: {config.numberOfEachQuery}\n")

    origRtreeQueriesTraining, origRtreeQueriesEvaluation = prepareRtree(config, origRtree, origTrajectories)

    compressionRateScores = list()

    #origTrajectoriesSize = sum(list(map(lambda T: len(T.nodes), origTrajectories)))

    ## Main Loop
    #print("Main loop..")

    # Sort compression_rate from highest to lowest
    giveQueryScorings(origRtree, origTrajectories, origRtreeQueriesTraining)
    simpTrajectories = dropNodes(origRtree, origTrajectories, config.compression_rate)

    simpRtree, simpTrajectories = loadRtree(SIMPLIFIEDDATABASENAME, simpTrajectories)

    compressionRateScores.append({ 'cr' : config.compression_rate, 'f1Scores' : getAverageF1ScoreAll(origRtreeQueriesEvaluation, origRtree, simpRtree), 'simplificationError' : GetSimplificationError(ORIGTrajectories, simpTrajectories), 'simplifiedTrajectories' : copy.deepcopy(simpTrajectories)}) #, GetSimplificationError(origTrajectories, simpTrajectories)
    # While above compression rate
    print(compressionRateScores[-1]['f1Scores'])
    simpRtree.close()

    if os.path.exists(SIMPLIFIEDDATABASENAME + '.data') and os.path.exists(SIMPLIFIEDDATABASENAME + '.index'):
        os.remove(SIMPLIFIEDDATABASENAME + '.data')
        os.remove(SIMPLIFIEDDATABASENAME + '.index')
    # Generate and apply queries, giving scorings to points

    # Remove x points with the fewest points

    # Collect evaluation data
        # getAverageF1ScoreAll, GetSimplificationError

    ## Save results
    with open(os.path.join(os.getcwd(), 'scores.pkl'), 'wb') as file:
        pickle.dump(compressionRateScores, file)
        file.close()

    ## Plot models

    pass


def gridSearch(allCombinations):
    configScore = list()
    for config in tqdm(allCombinations):
        origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)
        origRtreeQueriesTraining, origRtreeQueriesEvaluation = prepareRtree(config, origRtree, origTrajectories)

        ORIGTrajectories = copy.deepcopy(origTrajectories)

        giveQueryScorings(origRtree, origTrajectories, origRtreeQueriesTraining)
        logger.info(f'Processing config with epochs: {config.epochs}')
        simpTrajectories = dropNodes(origRtree, origTrajectories, config.compression_rate)

        logger.info('Loading simplified trajectories into Rtree.')
        simpRtree, simpTrajectories = loadRtree(SIMPLIFIEDDATABASENAME, simpTrajectories)

        f1score = getAverageF1ScoreAll(origRtreeQueriesEvaluation, origRtree, simpRtree)
        simplificationError = GetSimplificationError(ORIGTrajectories, simpTrajectories)
        
        configScore.append({
            'cr': config.compression_rate,
            'f1Scores': f1score,
            'simplificationError': simplificationError
        })

        logger.info('Run info: %s', configScore[-1]['f1Scores'])
        simpRtree.close()

        if os.path.exists(SIMPLIFIEDDATABASENAME + '.data') and os.path.exists(SIMPLIFIEDDATABASENAME + '.index'):
            os.remove(SIMPLIFIEDDATABASENAME + '.data')
            os.remove(SIMPLIFIEDDATABASENAME + '.index')

    ## Save results
    try:
        with open(os.path.join(os.getcwd(), 'scores.pkl'), 'wb') as file:
            pickle.dump(configScore, file)
            file.close()
    except Exception as e:
        print(f"err saving results: {e}")
        logger.error("problems when saving results to pickle file.")
        logger.error(traceback.format_exc())

    ## Plot models
    pass

if __name__ == "__main__":
    logger.info("---------------------------    Main Start    ---------------------------")
    # Create a single configuration object
    config = Configuration(
        epochs=100,
        compression_rate=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],  
        DB_size=100,
        trainTestSplit=0.8,
        numberOfEachQuery=100,
        QueriesPerTrajectory=0.005,
        verbose=True
    )

    try:
        # If compression_rate is a list, use gridSearch
        if isinstance(config.compression_rate, list):
            # Create all combinations for grid search
            allCombinations = createConfigs(
                [config.epochs],
                config.compression_rate,
                [config.DB_size],
                [config.trainTestSplit],
                [config.numberOfEachQuery],
                [config.QueriesPerTrajectory],
                [config.verbose]
            )
            gridSearch(allCombinations)
        else:
            # For single config testing
            main(config)
            
        print("Script finished successfully.") 

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

