from src.evaluation import getAverageF1ScoreAll, GetSimplificationError, getF1Score
from src.Util import ParamUtil
from src.QueryWrapper import QueryWrapper
from src.scoringQueries import giveQueryScorings
from src.load import build_Rtree, load_Tdrive, loadRtree, load_Tdrive_Rtree, get_Tdrive
from src.dropNodes import dropNodes
from src.Query import Query
from src.clusterQuery import ClusterQuery

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
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from rtree import index

sys.path.append("src/")

CSVNAME = 'trimmed_small_train'#first_10000_train'
DATABASENAME = 'original_Taxi'
SIMPLIFIEDDATABASENAME = 'simplified_Taxi'
PICKLE_HITS = ['RangeQueryHits.pkl'] # Define filenames for query hits
MULTIPROCESSTEST = True
PAGESIZE = 16000


def workerFnc(inputTuple):
    # Syntactic sugar
    queries = inputTuple[0]
    rtree, trajectories = loadRtree(CSVNAME+str(inputTuple[3]), inputTuple[1])[0], origTrajectories 

    # rtree, trajectories = build_Rtree("trimmed_small_train.csv", filename=("trimmed_small_train"+ str(inputTuple[3])))#get_Tdrive(filename="TDrive.csv")

    for query in queries:
        result = query.run(rtree)
        #print(result)
        if not isinstance(query, ClusterQuery):
            lock.acquire()
            query.distribute(trajectories, result)
            lock.release()
            #print(result[0].nodes[0].score)
        else:
            query.distribute(trajectories)
    print("Worker Final:", list(trajectories.items())[0][1].nodes[0].score)
    rtree.close()
    return trajectories

def f1ScoreWorkerFnc(inputTuple):
    out = []
    queries: list[Query] = inputTuple[0]
    rtree, trajectories = loadRtree(CSVNAME+str(inputTuple[3]), inputTuple[1])[0], simpTrajectories 
    simpRtree, _ = loadRtree(SIMPLIFIEDDATABASENAME+str(inputTuple[3]), simpTrajectories)

    for query in queries:
        val = getF1Score(query, rtree, simpRtree, trajectories)
        # print(val)
        out.append(val)
    # print(out)
    rtree.close()
    simpRtree.close()
    return sum(out)/len(out)


def prepare():
    ## Load Dataset
    #load_Tdrive(CSVNAME + '.csv', CSVNAME + '_trimmed.csv')

    #origRtree, origTrajectories = build_Rtree(CSVNAME + '_trimmed.csv', filename=DATABASENAME)
    logger.info('Starting get_Tdrive.')    

    global lock
    lock = mp.Lock()

    global origTrajectories
    origRtree, origTrajectories = build_Rtree("trimmed_small_train.csv", filename="trimmed_small_train")#get_Tdrive(filename="TDrive.csv")

    # print(origRtree.properties)
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
    if config["QueriesPerTrajectory"] != None : config["numberOfEachQuery"] = math.floor(config["QueriesPerTrajectory"] * len(origTrajectories.values()))


    # ---- Create training queries -----
    logger.info('Creating training queries.')
    origRtreeQueriesTraining : QueryWrapper = QueryWrapper(math.ceil(config["numberOfEachQuery"] * config["trainTestSplit"]))
    origRtreeParamsTraining : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours


    #logger.info('Creating range queries.')
    #origRtreeQueriesTraining.createRangeQueries(origRtree, origRtreeParamsTraining)
    #logger.info('Creating similarity queries.')
    #origRtreeQueriesTraining.createSimilarityQueries(origRtree, origRtreeParamsTraining)
    logger.info('Creating KNN queries.')
    origRtreeQueriesTraining.createKNNQueries(origRtree, origRtreeParamsTraining)
    # origRtreeQueriesTraining.createClusterQueries(origRtree, origRtreeParamsTraining)

    # ---- Create evaluation queries -----
    logger.info('Creating evaluation queries.')
    origRtreeQueriesEvaluation : QueryWrapper = QueryWrapper(math.floor(config["numberOfEachQuery"] - config["numberOfEachQuery"] * config["trainTestSplit"]))
    origRtreeParamsEvaluation : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours

    #logger.info('Creating range queries.')
    #origRtreeQueriesEvaluation.createRangeQueries(origRtree, origRtreeParamsEvaluation)
    #logger.info('Creating similarity queries.')
    #origRtreeQueriesEvaluation.createSimilarityQueries(origRtree, origRtreeParamsEvaluation)
    logger.info('Creating KNN queries.')
    origRtreeQueriesEvaluation.createKNNQueries(origRtree, origRtreeParamsEvaluation)
    # origRtreeQueriesEvaluation.createClusterQueries(origRtree, origRtreeParamsEvaluation)



    compressionRateScores = list()

    return origRtree, origTrajectories, origRtreeQueriesTraining, origRtreeQueriesEvaluation, ORIGTrajectories, compressionRateScores

    

#### main
def main(config):
    logger.info('Prepare data')
    origRtree, origTrajectories, origRtreeQueriesTraining, origRtreeQueriesEvaluation, ORIGTrajectories, compressionRateScores = prepare()

    # Sort compression_rate from highest to lowest
    #origTrajectoriesSize = sum(list(map(lambda T: len(T.nodes), origTrajectories)))

    ## Main Loop
    #print("Main loop..")
    if MULTIPROCESSTEST:
        # Declare worker input list
        inputList = []

        # Get all the lists of queries
        queryList: list[list[Query]] = origRtreeQueriesTraining.getQueries()

        # Amount of each query loaded
        numEachQueryType = math.floor((config["numberOfEachQuery"]*config["trainTestSplit"]))

        # Amount of query types loaded
        numQueryTypes = math.floor(len(queryList)/numEachQueryType)

        # Compile the list of tuples containing queries of each type for each process to evaluate
        # along with the RTree and the original trajectories
        for i in range(os.cpu_count()):
            processQueries = []
            for j in range(numQueryTypes):
                # Find the amount of queries to *ideally* distribute to each process
                splitLength = math.ceil(numEachQueryType/os.cpu_count())

                # print(splitLength)

                firstQueryIdx = (i * splitLength + (j * numEachQueryType))*numQueryTypes
                lastQueryIdx = ((i+1) * splitLength + (j * numEachQueryType))*numQueryTypes

                if i == os.cpu_count() - 1:
                    processQueries.extend(queryList[firstQueryIdx:-1])
                    continue

                processQueries.extend(queryList[firstQueryIdx:lastQueryIdx])

            inputList.append([processQueries, origTrajectories, origRtree, i])
        
        # for i, chunk in enumerate(inputList):
        #     print(f"Process {i} received {len(chunk[0])} queries")
        
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor: #os.cpu_count()
            # Run the worker function
            res = list(executor.map(workerFnc, inputList))
            executor.shutdown(wait=True)

        combined = defaultdict(lambda: copy.deepcopy(list(res[0].values())[0]))

        for r in res:
            for tid, traj in r.items():
                if tid not in combined:
                    combined[tid] = traj
                else:
                    for i in range(len(traj.nodes)):
                        combined[tid].nodes[i].score += traj.nodes[i].score

        # then sum scores
        score = sum(node.score for traj in combined.values() for node in traj.nodes)

        # for key in res[11].keys():
        #     for i in range(len(res[0][key].nodes)):
        #         score += res[11][key].nodes[i].score + res[1][key].nodes[i].score + res[2][key].nodes[i].score
        print()
        print(len(res))
        print(score)
        print()

        # for result in res[1:]:
        #     print(result)
    # for tid, traj in res[0].items():
    #     print("Parent See:", traj.nodes[0].score)

    for key in origTrajectories.keys():
        for i in range(len(res)):
            for node in origTrajectories[key].nodes:
                node.score += res[i][key].nodes[node.id].score
    # print(score)
    # # Sort compression_rate from highest to lowest
    config["compression_rate"].sort(reverse=True)
    if MULTIPROCESSTEST:
        # Get all the lists of queries
        queryList: list[list[Query]] = origRtreeQueriesEvaluation.getQueries()

        # Amount of each query loaded
        numEachQueryType = math.floor((config["numberOfEachQuery"]*config["trainTestSplit"]))

        # Amount of query types loaded
        numQueryTypes = math.floor(len(queryList)/numEachQueryType)

        # Declare worker input list
        inputList = []

        # Compile the list of tuples containing queries of each type for each process to evaluate
        # along with the RTree and the original trajectories
        for i in range(os.cpu_count()):
            processQueries = []
            for j in range(numQueryTypes):
                # Find the amount of queries to *ideally* distribute to each process
                splitLength = math.ceil(numEachQueryType/os.cpu_count())

                # print(splitLength)

                firstQueryIdx = (i * splitLength + (j * numEachQueryType))*numQueryTypes
                lastQueryIdx = ((i+1) * splitLength + (j * numEachQueryType))*numQueryTypes

                if i == os.cpu_count() - 1:
                    processQueries.extend(queryList[firstQueryIdx:-1])
                    continue

                processQueries.extend(queryList[firstQueryIdx:lastQueryIdx])
            #print(processQueries)
            inputList.append([processQueries])
        # print(inputList)

    # Begin evaluation at different compression rates
    # giveQueryScorings(origRtree, origTrajectories, origRtreeQueriesTraining)
    for cr in tqdm(config["compression_rate"], desc="compression rate"):
        if MULTIPROCESSTEST:
            print(f"\nCompression rate: {cr}")
            global simpTrajectories
            simpTrajectories = dropNodes(origRtree, origTrajectories, cr)
            global simpRtree
            simpRtree, simpTrajectories = loadRtree(SIMPLIFIEDDATABASENAME, simpTrajectories)
            
            id = 0
            for input in inputList:
                input.append(simpTrajectories)
                input.append(simpRtree)
                input.append(id)
                id += 1

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor: #os.cpu_count()
                # Run the worker function
                res = list(executor.map(f1ScoreWorkerFnc, inputList))
                executor.shutdown(wait=True)

            # print(res)
                print()
                print(res)
                print()
            averageF1Score = sum(res)/len(res)

            compressionRateScores.append({ 'cr' : cr, 'f1Score' : averageF1Score, 'simplificationError' : GetSimplificationError(ORIGTrajectories, simpTrajectories)}) #, GetSimplificationError(origTrajectories, simpTrajectories)
            # While above compression rate
            print()
            print(compressionRateScores[-1]['f1Score'])
            print()

            simpRtree.close()
        elif not MULTIPROCESSTEST:
            logger.info('Give nodes scores.')
            giveQueryScorings(origRtree, origTrajectories, origRtreeQueriesTraining)

            # Begin evaluation at different compression rates


            logger.info('Performing loop for compression rate %s', cr);

            logger.info('Dropping nodes.')
            simpTrajectories = dropNodes(origRtree, origTrajectories, cr)

            logger.info('Loading simplified trajectories into Rtree.')
            simpRtree, simpTrajectories = loadRtree(SIMPLIFIEDDATABASENAME, simpTrajectories)

            logger.info('Calculating f1 scores.')
            f1score = getAverageF1ScoreAll(origRtreeQueriesEvaluation, origRtree, simpRtree, ORIGTrajectories)
            logger.info('Calculating simplification errors.')
            simplificationError = GetSimplificationError(ORIGTrajectories, simpTrajectories)
            compressionRateScores.append({ 'cr' : cr, 'f1Scores' : f1score, 'simplificationError' : simplificationError})
            # While above compression rate
            logger.info('Run info: %s',compressionRateScores[-1]['f1Scores'])
            simpRtree.close()

        if os.path.exists(SIMPLIFIEDDATABASENAME + '.data') and os.path.exists(SIMPLIFIEDDATABASENAME + '.index'):
            os.remove(SIMPLIFIEDDATABASENAME + '.data')
            os.remove(SIMPLIFIEDDATABASENAME + '.index')
        # Generate and apply queries, giving scorings to points

        # Remove x points with the fewest points

        # Collect evaluation data
            # getAverageF1ScoreAll, GetSimplificationError

    ## Save results
    try:
        with open(os.path.join(os.getcwd(), 'scores.pkl'), 'wb') as file:
            pickle.dump(compressionRateScores, file)
            file.close()
    except Exception as e:
        print(f"err saving results: {e}")
        logger.error("problems when saving results to pickle file.")
        logger.error(traceback.format_exc())

    ## Plot models

    pass




if __name__ == "__main__":
    logger.info("---------------------------    Main Start    ---------------------------")
    config = {}
    config["epochs"] = 100                  # Number of epochs to simplify the trajectory database
    config["compression_rate"] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]      # Compression rate of the trajectory database
    config["DB_size"] = 100                 # Amount of trajectories to load (Potentially irrelevant)
    config["verbose"] = True                # Print progress
    config["trainTestSplit"] = 0.8          # Train/test split
    config["numberOfEachQuery"] = 100       # Number of queries used to simplify database    
    config["QueriesPerTrajectory"] = None   # Number of queries per trajectory, in percentage. Overrides numberOfEachQuery if not none

    #print("Script starting...")
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
