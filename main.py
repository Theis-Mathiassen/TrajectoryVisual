from src.gridSearch import createConfigs, Configuration, Weights
from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
from src.Util import ParamUtil
from src.QueryWrapper import QueryWrapper
from src.scoringQueries import giveQueryScorings
from src.load import build_Rtree, load_Tdrive, loadRtree, load_Tdrive_Rtree, get_Tdrive
from src.dropNodes import dropNodes
from src.log import logger, ERROR_LOG_FILENAME
from tqdm import tqdm
from pathlib import Path
import os
import sys
import copy
import pickle
import math
import os
import traceback # traceback for information on python stack traces
import argparse
from dataclasses import asdict

import numpy as np

sys.path.append("src/")
output_dir = os.environ.get('JOB_OUTPUT_DIR', os.getcwd())
Path(output_dir).mkdir(parents=True, exist_ok=True)


DATABASENAME = 'original_Taxi'
SIMPLIFIEDDATABASENAME = 'simplified_Taxi'
PICKLE_HITS = []#['RangeQueryDataHits.pkl', 'KnnQueryDataHits.pkl', 'SimilarityQueryDataHits.pkl'] #['RangeQueryHits_geolife.pkl'] 
CACHE_FILE = os.path.join(output_dir, 'cached_rtree_query_eval_results.pkl')
TEMPORAL_WINDOW = 10800
SPATIAL_WINDOW = 2000

# Prepare RTrees for training and testing
def prepareQueries(config, origRtree, origTrajectories, useGaussian = False, useDataDistribution = False):

    # ---- 
    avgCoordinateValues = None
    stdCoordinatesValue = None
    if useGaussian: 
        avgCoordinatesValue = getAverageNodeCoordinates(origTrajectories)
        stdCoordinatesValue = getStandardDerivationNodeCoordinates(origTrajectories, avgCoordinatesValue)
        print(avgCoordinatesValue)
        print(stdCoordinatesValue)
    elif useDataDistribution:
        dataTrajGrid, dataNodeGrid = getDataDistribution(origTrajectories, SPATIAL_WINDOW, TEMPORAL_WINDOW)
        listTrajCellTup = list(dataTrajGrid.keys())
        #listTrajCell = np.array(list(dataTrajGrid.values()))
        listTrajCellCnt = np.array([len(cell) for cell in list(dataTrajGrid.values())])
        totalTrajCellCnt = np.sum(listTrajCellCnt)
        #listTrajCellProbDist = list
        #sampleTrajCellDist = 
        
        keys = np.random.choice(len(listTrajCellTup), 10000, p=(listTrajCellCnt/totalTrajCellCnt))
        listSampleCellKeys = [listTrajCellTup[key] for key in keys]
        print(len(set(listSampleCellKeys)))

    # ---- Create training queries -----
    logger.info('Creating training queries.')
    origRtreeQueriesTraining : QueryWrapper = QueryWrapper(math.ceil(config.numberOfEachQuery * config.trainTestSplit), random=False, useGaussian=useGaussian, avgCoordinateValues=avgCoordinateValues, rtree=origRtree, sigma=stdCoordinatesValue)
    origRtreeParamsTraining : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=TEMPORAL_WINDOW) # Temporal window for T-Drive is 3 hours

    
    if useDataDistribution:
        logger.info('Creating range queries.')
        origRtreeQueriesTraining.createRangeQueries(origRtree, origRtreeParamsTraining, flag=config.range_flag, cellDist=listSampleCellKeys)
        logger.info('Creating similarity queries.')
        origRtreeQueriesTraining.createSimilarityQueries(origRtree, origRtreeParamsTraining, scoring_system=config.similarity_system, cellDist=listSampleCellKeys, trajGrid=dataTrajGrid)
        logger.info('Creating KNN queries.')
        origRtreeQueriesTraining.createKNNQueries(origRtree, origRtreeParamsTraining, distance_method=config.knn_method, cellDist=listSampleCellKeys, trajGrid=dataTrajGrid)
        #origRtreeQueriesTraining.createClusterQueries(origRtree, origRtreeParamsTraining)
    else:
        logger.info('Creating range queries.')
        origRtreeQueriesTraining.createRangeQueries(origRtree, origRtreeParamsTraining, flag=config.range_flag)
        logger.info('Creating similarity queries.')
        origRtreeQueriesTraining.createSimilarityQueries(origRtree, origRtreeParamsTraining, scoring_system=config.similarity_system)
        logger.info('Creating KNN queries.')
        origRtreeQueriesTraining.createKNNQueries(origRtree, origRtreeParamsTraining, distance_method=config.knn_method)
        #logger.info('Creating cluster queries')
        #origRtreeQueriesTraining.createClusterQueries(origRtree, origRtreeParamsTraining)

    # ---- Create evaluation queries -----
    logger.info('Creating evaluation queries.')
    origRtreeQueriesEvaluation : QueryWrapper = QueryWrapper(math.floor(config.numberOfEachQuery - config.numberOfEachQuery * config.trainTestSplit),random=False, rtree=origRtree)
    origRtreeParamsEvaluation : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=TEMPORAL_WINDOW) # Temporal window for T-Drive is 3 hours

    if useDataDistribution:
        logger.info('Creating range queries.')
        origRtreeQueriesEvaluation.createRangeQueries(origRtree, origRtreeParamsEvaluation, flag=config.range_flag, cellDist=listSampleCellKeys)
        logger.info('Creating similarity queries.')
        origRtreeQueriesEvaluation.createSimilarityQueries(origRtree, origRtreeParamsEvaluation, scoring_system=config.similarity_system, cellDist=listSampleCellKeys, trajGrid = dataTrajGrid)
        logger.info('Creating KNN queries.')
        origRtreeQueriesEvaluation.createKNNQueries(origRtree, origRtreeParamsEvaluation, distance_method=config.knn_method, cellDist=listSampleCellKeys, trajGrid = dataTrajGrid)
        #logger.info('Creating cluster queries.')
        #origRtreeQueriesEvaluation.createClusterQueries(origRtree, origRtreeParamsEvaluation, cellDist=listSampleCellKeys, trajGrid= dataTrajGrid)
    else:
        logger.info('Creating range queries.')
        origRtreeQueriesEvaluation.createRangeQueries(origRtree, origRtreeParamsEvaluation, flag=config.range_flag)
        logger.info('Creating similarity queries.')
        origRtreeQueriesEvaluation.createSimilarityQueries(origRtree, origRtreeParamsEvaluation, scoring_system=config.similarity_system)
        logger.info('Creating KNN queries.')
        origRtreeQueriesEvaluation.createKNNQueries(origRtree, origRtreeParamsEvaluation, distance_method=config.knn_method)
    # origRtreeQueriesEvaluation.createClusterQueries(origRtree, origRtreeParamsEvaluation)
    return origRtreeQueriesTraining, origRtreeQueriesEvaluation


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

#### main
def main(config):
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        logger.info(f"Cleared cache file: {CACHE_FILE}")

    logger.info('Starting get_Tdrive.')
    origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)
    logger.info('Completed get_Tdrive.')

    logger.info('Copying trajectories.')
    uncompressedTrajectories = {
        tid : copy.deepcopy(traj)
        for tid, traj, in tqdm(origTrajectories.items(), desc = "Copying trajectories")
    }

    if config.QueriesPerTrajectory is not None:
        config.numberOfEachQuery = math.floor(config.QueriesPerTrajectory * len(origTrajectories.values()))
    logger.info(f"Number of queries to be created: {config.numberOfEachQuery}")

    #origRtreeQueriesTraining, origRtreeQueriesEvaluation = prepareQueries(config, origRtree, origTrajectories, useGaussian=True)
    origRtreeQueriesTraining, origRtreeQueriesEvaluation = prepareQueries(config, origRtree, origTrajectories, useGaussian=False, useDataDistribution=True)

    giveQueryScorings(origRtree, origTrajectories, origRtreeQueriesTraining, pickleFiles=PICKLE_HITS, config=config)
    #giveQueryScorings(origRtree, origTrajectories, origRtreeQueriesTraining, pickleFiles=None, config=config)

    simpTrajectories = dropNodes(origRtree, origTrajectories, config.compression_rate)
    simpRtree, simpTrajectories = loadRtree(SIMPLIFIEDDATABASENAME, simpTrajectories)

    compressionRateScores = list()
    compressionRateScores.append({ 'cr' : config.compression_rate, 'f1Scores' : getAverageF1ScoreAll(origRtreeQueriesEvaluation, origRtree, simpRtree, uncompressedTrajectories), 'simplificationError' : GetSimplificationError(uncompressedTrajectories, simpTrajectories), 'simplifiedTrajectories' : copy.deepcopy(simpTrajectories)}) #, GetSimplificationError(origTrajectories, simpTrajectories)
    print(compressionRateScores[-1]['f1Scores'])

    simpRtree.close()

    if os.path.exists(os.path.join(output_dir, SIMPLIFIEDDATABASENAME + '.data')) and os.path.exists(os.path.join(output_dir, SIMPLIFIEDDATABASENAME + '.index')):
        os.remove(os.path.join(output_dir, SIMPLIFIEDDATABASENAME + '.data'))
        os.remove(os.path.join(output_dir, SIMPLIFIEDDATABASENAME + '.index'))

    ## Save results
    with open(os.path.join(output_dir, 'scores.pkl'), 'wb') as file:
        pickle.dump(compressionRateScores, file)
        file.close()


def gridSearch(config, args):
    configScore = list()
    # we need to clear cache file for each configuration
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        logger.info(f"Cleared cache file: {CACHE_FILE}")

    origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)
    
    #origRtreeQueriesTraining, origRtreeQueriesEvaluation = prepareQueries(config, origRtree, origTrajectories, useGaussian=True)
    origRtreeQueriesTraining, origRtreeQueriesEvaluation = prepareQueries(config, origRtree, origTrajectories, useGaussian=False, useDataDistribution=True)

    uncompressedTrajectories = copy.deepcopy(origTrajectories)

    giveQueryScorings(origRtree, origTrajectories, queryWrapper = origRtreeQueriesTraining, pickleFiles=PICKLE_HITS, config=config, numberToTrain=400)
    for weight in config.weights:
        weight = asdict(weight) # convert to dict
        for compression_rate in config.compression_rate:
            simpTrajectories = dropNodes(origRtree, origTrajectories, compression_rate, weights=weight)
            logger.info('Loading simplified trajectories into Rtree.')
            simpRtree, simpTrajectories = loadRtree(SIMPLIFIEDDATABASENAME, simpTrajectories)
            f1score = getAverageF1ScoreAll(origRtreeQueriesEvaluation, origRtree, simpRtree, uncompressedTrajectories)
            simplificationError = GetSimplificationError(uncompressedTrajectories, simpTrajectories)

            #giveQueryScorings(origRtree, origTrajectories, queryWrapper = origRtreeQueriesTraining, pickleFiles=None, config=config)


            configScore.append({
                'cr': compression_rate,
                'weights': weight,
                'f1Scores': f1score,
                'simplificationError': simplificationError
            })
            logger.info('Run info: %s, weights: %s, compression rate: %s', configScore[-1]['f1Scores'], configScore[-1]['weights'], configScore[-1]['cr'])

            simpRtree.close()

            if os.path.exists(os.path.join(output_dir, SIMPLIFIEDDATABASENAME + '.data')) and os.path.exists(os.path.join(output_dir, SIMPLIFIEDDATABASENAME + '.index')):
                os.remove(os.path.join(output_dir, SIMPLIFIEDDATABASENAME + '.data'))
                os.remove(os.path.join(output_dir, SIMPLIFIEDDATABASENAME + '.index'))

    ## Save results with unique filename based on the combo of query methods through cmd args
    try:
        filename = f"scores_knn{args.knn}_range{args.range}_sim{args.similarity}.pkl"
        with open(os.path.join(output_dir, filename), 'wb') as file:
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
    print(f"Using query methods - KNN: {args.knn}, Range: {args.range}, Similarity: {args.similarity}")
    
    # Create a single configuration object
    config = Configuration(
        compression_rate=[0.9, 0.95],
        DB_size=100,
        trainTestSplit=0.8,
        numberOfEachQuery=500,
        QueriesPerTrajectory=None,
        verbose=True,
        knn_method=args.knn,
        range_flag=args.range,
        similarity_system=args.similarity,
        # weights = [{'range' : 1,  'knn' : 0.25, 'similarity' : 1, 'cluster' : 0},
        #            {'range' : 1,  'knn' : 1, 'similarity' : 0.5, 'cluster' : 0},
        #            {'range' : 0.5,  'knn' : 1, 'similarity' : 1, 'cluster' : 0},
        #            {'range' : 1,  'knn' : 1, 'similarity' : 2, 'cluster' : 0}]
        weights = [Weights(range=1, similarity=1, knn=1, cluster=1)]
        
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
                [config.similarity_system],
                [config.weights]
            )
            gridSearch(config, args)
        else:
            # For single config testing
            print("Not running grid search")
            main(config)
            
        print("Script finished successfully.") 

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
        filePath = os.path.join(output_dir, fileString)
        if os.path.exists(filePath):
            os.remove(filePath)
