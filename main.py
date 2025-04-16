from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
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
import logging
import traceback # traceback for information on python stack traces
import multiprocessing as mp
from collections import defaultdict

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

def initPoolProcesses(the_lock, the_trajectories):
    global lock
    global trajectories
    lock = the_lock
    trajectories = the_trajectories

def workerFnc(inputTuple):
    # Syntactic sugar
    queries = inputTuple[0]
    rtree = inputTuple[1]

    for query in queries:
        result = query.run(rtree)
        if not isinstance(query, ClusterQuery):
            lock.acquire()
            query.distribute(trajectories, result)
            lock.release()
        else:
            lock.acquire()
            query.distribute(trajectories)
            lock.release()
    return trajectories

#### main
def main(config):
    ## Load Dataset
    #load_Tdrive(CSVNAME + '.csv', CSVNAME + '_trimmed.csv')
    
    #origRtree, origTrajectories = build_Rtree(CSVNAME + '_trimmed.csv', filename=DATABASENAME)

    origRtree, origTrajectories = build_Rtree("trimmed_small_train.csv", filename="trimmed_small_train")#get_Tdrive(filename="TDrive.csv")

    # print(origRtree.properties)

    # simpRtree, simpTrajectories = build_Rtree("first_10000_train_trimmed.csv", filename="simplified_Taxi")
    ## Setup reinforcement learning algorithms (t2vec, etc.)

    ORIGTrajectories = copy.deepcopy(origTrajectories)

    ## Setup data collection environment, that is evaluation after each epoch

    # ---- Set number of queries to be created ----
    if config["QueriesPerTrajectory"] != None : config["numberOfEachQuery"] = math.floor(config["QueriesPerTrajectory"] * len(origTrajectories.values()))

    print(f"\n\nNumber of queries to be created: {config['numberOfEachQuery']}\n")

    # ---- Create training queries -----
    origRtreeQueriesTraining : QueryWrapper = QueryWrapper(math.ceil(config["numberOfEachQuery"] * config["trainTestSplit"]))
    origRtreeParamsTraining : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours
    

    origRtreeQueriesTraining.createRangeQueries(origRtree, origRtreeParamsTraining)
    origRtreeQueriesTraining.createSimilarityQueries(origRtree, origRtreeParamsTraining)
    origRtreeQueriesTraining.createKNNQueries(origRtree, origRtreeParamsTraining)
    # origRtreeQueriesTraining.createClusterQueries(origRtree, origRtreeParamsTraining)

    # ---- Create evaluation queries -----
    origRtreeQueriesEvaluation : QueryWrapper = QueryWrapper(math.floor(config["numberOfEachQuery"] - config["numberOfEachQuery"] * config["trainTestSplit"]))
    origRtreeParamsEvaluation : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours

    origRtreeQueriesEvaluation.createRangeQueries(origRtree, origRtreeParamsEvaluation)
    origRtreeQueriesEvaluation.createSimilarityQueries(origRtree, origRtreeParamsEvaluation)
    origRtreeQueriesEvaluation.createKNNQueries(origRtree, origRtreeParamsEvaluation)
    # origRtreeQueriesEvaluation.createClusterQueries(origRtree, origRtreeParamsEvaluation)


    
    compressionRateScores = list()

    #origTrajectoriesSize = sum(list(map(lambda T: len(T.nodes), origTrajectories)))

    ## Main Loop
    #print("Main loop..")
    # Declare worker input list
    inputList = []

    # Get all the lists of queries
    queryList: list[list[Query]] = origRtreeQueriesTraining.getQueries()

    # Amount of each query loaded
    numEachQueryType = math.floor((config["numberOfEachQuery"]*config["trainTestSplit"]))

    # Amount of query types loaded
    numQueryTypes = math.floor(len(queryList)/numEachQueryType)

    # print(numQueryTypes)

    ##print(queryList)

    # Compile the list of tuples containing queries of each type for each process to evaluate
    # along with the RTree and the original trajectories
    for i in range(os.cpu_count()):
        processQueries = []
        for j in range(numQueryTypes):
            # Find the amount of queries to *ideally* distribute to each process
            splitLength = math.ceil(numEachQueryType/os.cpu_count())

            #print(splitLength)

            firstQueryIdx = i * splitLength + (j * numEachQueryType)
            lastQueryIdx = (i+1) * splitLength + (j * numEachQueryType)
            #print(firstQueryIdx, lastQueryIdx)
            
            # print(queryList[firstQueryIdx:lastQueryIdx])

            if i == os.cpu_count() - 1:
                processQueries.extend(queryList[(i * splitLength + j * numEachQueryType):-1])
                continue

            processQueries.extend(queryList[firstQueryIdx:lastQueryIdx])
        # print(processQueries)

        inputList.append([processQueries, origRtree])

    for i, chunk in enumerate(inputList):
        print(f"Process {i} received {len(chunk[0])} queries")

    #inputList.append((queryList[((os.cpu_count()-1)*splitLength):-1], origRtree, origTrajectories))
    
    # print(inputList[0])
    #exit()

    lock = mp.Lock()
    with mp.Pool(processes=os.cpu_count(), initializer=initPoolProcesses, initargs=(lock, origTrajectories)) as pool: #os.cpu_count()
        # Run the worker function
        res = pool.map(workerFnc, inputList)

        # Wait for all processes to complete
        pool.close()
        pool.join()

    # print(len(res))
    # print(res[0])
    # for val in zip(res[0], res[1], res[2]):
    #     print(val[0], val[1], val[2])
    # print(res[0][1372636858620000589].nodes[0].score)
    # print(res[1][1372636858620000589].nodes[0].score)
    # print(res[2][1372636858620000589].nodes[0].score)
    # print(res[0][1372644436620000112].nodes[0].score)
    # print(res[1][1372644436620000112].nodes[0].score)
    # print(res[2][1372644436620000112].nodes[0].score)

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
    print(len(res))
    print(score)
    # for result in res[1:]:
    #     print(result)
    exit()

    # Sort compression_rate from highest to lowest
    config["compression_rate"].sort(reverse=True)
    giveQueryScorings(origRtree, origTrajectories, origRtreeQueriesTraining)
    for cr in tqdm(config["compression_rate"], desc="compression rate"):        
        simpTrajectories = dropNodes(origRtree, origTrajectories, cr)

        simpRtree, simpTrajectories = loadRtree(SIMPLIFIEDDATABASENAME, simpTrajectories)

        compressionRateScores.append({ 'cr' : cr, 'f1Scores' : getAverageF1ScoreAll(origRtreeQueriesEvaluation, origRtree, simpRtree), 'simplificationError' : GetSimplificationError(ORIGTrajectories, simpTrajectories), 'simplifiedTrajectories' : copy.deepcopy(simpTrajectories)}) #, GetSimplificationError(origTrajectories, simpTrajectories)
        # While above compression rate
        print()
        print(compressionRateScores[-1]['f1Scores'])
        print()
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
    config = {}
    config["epochs"] = 100                  # Number of epochs to simplify the trajectory database
    config["compression_rate"] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]      # Compression rate of the trajectory database
    config["DB_size"] = 100                 # Amount of trajectories to load (Potentially irrelevant)
    config["verbose"] = True                # Print progress
    config["trainTestSplit"] = 0.8          # Train/test split
    config["numberOfEachQuery"] = 100     # Number of queries used to simplify database    
    config["QueriesPerTrajectory"] = None   # Number of queries per trajectory, in percentage. Overrides numberOfEachQuery if not none

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