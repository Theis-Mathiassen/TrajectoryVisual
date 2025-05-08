from src.gridSearch import createConfigs, Configuration
from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
from src.Util import ParamUtil
from src.QueryWrapper import QueryWrapper
from src.scoringQueries import giveQueryScorings
from src.load import build_Rtree, load_Tdrive, loadRtree, load_Tdrive_Rtree, get_Tdrive
from src.dropNodes import dropNodes
from src.log import logger, ERROR_LOG_FILENAME
from tqdm import tqdm

import os
import sys
import copy
import pickle
import math
import os
import traceback # traceback for information on python stack traces
import argparse

sys.path.append("src/")

DATABASENAME = 'original_Taxi'
SIMPLIFIEDDATABASENAME = 'simplified_Taxi'
PICKLE_HITS = ['RangeQueryHits.pkl', 'KnnQueryHits.pkl', 'SimilarityQueryHits.pkl'] 
CACHE_FILE = 'cached_rtree_query_eval_results.pkl'

# Prepare RTrees for training and testing
def prepareRtree(config, origRtree, origTrajectories):
    # ---- Create training queries -----
    logger.info('Creating training queries.')
    origRtreeQueriesTraining : QueryWrapper = QueryWrapper(math.ceil(config.numberOfEachQuery * config.trainTestSplit))
    origRtreeParamsTraining : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours

    logger.info('Creating range queries.')
    origRtreeQueriesTraining.createRangeQueries(origRtree, origRtreeParamsTraining, flag=config.range_flag)
    logger.info('Creating similarity queries.')
    origRtreeQueriesTraining.createSimilarityQueries(origRtree, origRtreeParamsTraining, scoring_system=config.similarity_system)
    logger.info('Creating KNN queries.')
    origRtreeQueriesTraining.createKNNQueries(origRtree, origRtreeParamsTraining, distance_method=config.knn_method)
    # origRtreeQueriesTraining.createClusterQueries(origRtree, origRtreeParamsTraining)

    # ---- Create evaluation queries -----
    logger.info('Creating evaluation queries.')
    origRtreeQueriesEvaluation : QueryWrapper = QueryWrapper(math.floor(config.numberOfEachQuery - config.numberOfEachQuery * config.trainTestSplit))
    origRtreeParamsEvaluation : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours

    logger.info('Creating range queries.')
    origRtreeQueriesEvaluation.createRangeQueries(origRtree, origRtreeParamsEvaluation, flag=config.range_flag)
    logger.info('Creating similarity queries.')
    origRtreeQueriesEvaluation.createSimilarityQueries(origRtree, origRtreeParamsEvaluation, scoring_system=config.similarity_system)
    logger.info('Creating KNN queries.')
    origRtreeQueriesEvaluation.createKNNQueries(origRtree, origRtreeParamsEvaluation, distance_method=config.knn_method)
    # origRtreeQueriesEvaluation.createClusterQueries(origRtree, origRtreeParamsEvaluation)
    return origRtreeQueriesTraining, origRtreeQueriesEvaluation



#### main
def main(config):
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        logger.info(f"Cleared cache file: {CACHE_FILE}")

    logger.info('Starting get_Tdrive.')
    origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)
    logger.info('Completed get_Tdrive.')

    logger.info('Copying trajectories.')
    ORIGTrajectories = {
        tid : copy.deepcopy(traj)
        for tid, traj, in tqdm(origTrajectories.items(), desc = "Copying trajectories")
    }

    if config.QueriesPerTrajectory != None:
        config.numberOfEachQuery = math.floor(config.QueriesPerTrajectory * len(origTrajectories.values()))
    logger.info(f"Number of queries to be created: {config.numberOfEachQuery}")

    origRtreeQueriesTraining, origRtreeQueriesEvaluation = prepareRtree(config, origRtree, origTrajectories)

    giveQueryScorings(origRtree, origTrajectories, origRtreeQueriesTraining, pickleFiles=PICKLE_HITS)

    simpTrajectories = dropNodes(origRtree, origTrajectories, config.compression_rate)
    simpRtree, simpTrajectories = loadRtree(SIMPLIFIEDDATABASENAME, simpTrajectories)

    compressionRateScores = list()
    compressionRateScores.append({ 'cr' : config.compression_rate, 'f1Scores' : getAverageF1ScoreAll(origRtreeQueriesEvaluation, origRtree, simpRtree, origTrajectories), 'simplificationError' : GetSimplificationError(ORIGTrajectories, simpTrajectories), 'simplifiedTrajectories' : copy.deepcopy(simpTrajectories)}) #, GetSimplificationError(origTrajectories, simpTrajectories)
    print(compressionRateScores[-1]['f1Scores'])

    simpRtree.close()

    if os.path.exists(SIMPLIFIEDDATABASENAME + '.data') and os.path.exists(SIMPLIFIEDDATABASENAME + '.index'):
        os.remove(SIMPLIFIEDDATABASENAME + '.data')
        os.remove(SIMPLIFIEDDATABASENAME + '.index')

    ## Save results
    with open(os.path.join(os.getcwd(), 'scores.pkl'), 'wb') as file:
        pickle.dump(compressionRateScores, file)
        file.close()


def gridSearch(allCombinations, args):
    configScore = list()
    for config in tqdm(allCombinations):
        # we need to clear cache file for each configuration
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            logger.info(f"Cleared cache file: {CACHE_FILE}")

        origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)
        origRtreeQueriesTraining, origRtreeQueriesEvaluation = prepareRtree(config, origRtree, origTrajectories)

        ORIGTrajectories = copy.deepcopy(origTrajectories)

        giveQueryScorings(origRtree, origTrajectories, origRtreeQueriesTraining, pickleFiles=PICKLE_HITS)
        simpTrajectories = dropNodes(origRtree, origTrajectories, config.compression_rate)

        logger.info('Loading simplified trajectories into Rtree.')
        simpRtree, simpTrajectories = loadRtree(SIMPLIFIEDDATABASENAME, simpTrajectories)

        f1score = getAverageF1ScoreAll(origRtreeQueriesEvaluation, origRtree, simpRtree, origTrajectories)
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

    ## Save results with unique filename based on the combo of query methods through cmd args
    try:
        filename = f"scores_knn{args.knn}_range{args.range}_sim{args.similarity}.pkl"
        with open(os.path.join(os.getcwd(), filename), 'wb') as file:
            pickle.dump(configScore, file)
            file.close()
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        print(f"err saving results: {e}")
        logger.error("problems when saving results to pickle file.")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    logger.info("---------------------------    Main Start    ---------------------------")
    
    # Parse command line arguments
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Specify the method for distributing points.')
    parser.add_argument('--knn', type=int, help='Distance method 1 -> Use spatio temporal linear combine distance.')
    parser.add_argument('--range', type=int, help='Method: 1 -> winner_takes_all, 2 -> shared_equally(1 point total), 3 -> shared_equally(1 point each.), 4 -> gradient_points')
    parser.add_argument('--similarity', type=str, help='c -> Closest, a -> All, c+f -> Closest + Farthest, m -> moving away for a longer period than streak')
    args = parser.parse_args()
    logger.info(f"Using query methods - KNN: {args.knn}, Range: {args.range}, Similarity: {args.similarity}")
    
    # Create a single configuration object
    config = Configuration(
        compression_rate=[0.8, 0.9, 0.95, 0.975, 0.99],
        DB_size=100,
        trainTestSplit=0,
        numberOfEachQuery=100,
        QueriesPerTrajectory=None,
        verbose=True,
        knn_method=args.knn,
        range_flag=args.range,
        similarity_system=args.similarity
    )

    try:
        # If compression_rate is a list, use gridSearch
        if isinstance(config.compression_rate, list):
            print("Running grid search")
            # Create all combinations for grid search
            allCombinations = createConfigs(
                config.compression_rate,
                [config.DB_size],
                [config.trainTestSplit],
                [config.numberOfEachQuery],
                [config.QueriesPerTrajectory],
                [config.verbose],
                [config.knn_method],
                [config.range_flag],
                [config.similarity_system]
            )
            gridSearch(allCombinations, args)
        else:
            # For single config testing
            print("Not running grid search")
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

