from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
from src.Util import ParamUtil
from src.QueryWrapper import QueryWrapper
from src.scoringQueries import giveQueryScorings
from src.load import build_Rtree, load_Tdrive, loadRtree, load_Tdrive_Rtree
from src.dropNodes import dropNodes

import os
import sys
import copy
import pickle
import math
from tqdm import tqdm
import pandas as pd 
import os

sys.path.append("src/")

CSVNAME = 'first_10000_train'
DATABASENAME = 'original_Taxi'
SIMPLIFIEDDATABASENAME = 'simplified_Taxi'

#### main
def main(config):
    ## Load Dataset
    #load_Tdrive(CSVNAME + '.csv', CSVNAME + '_trimmed.csv')
    
    #origRtree, origTrajectories = build_Rtree(CSVNAME + '_trimmed.csv', filename=DATABASENAME)

    origRtree, origTrajectories = load_Tdrive_Rtree(filename=DATABASENAME)

    #simpRtree, simpTrajectories = build_Rtree("first_10000_train_trimmed.csv", filename="simplified_Taxi")
    ## Setup reinforcement learning algorithms (t2vec, etc.)

    ORIGTrajectories = copy.deepcopy(origTrajectories)

    ## Setup data collection environment, that is evaluation after each epoch
    # ---- Create training queries -----
    origRtreeQueriesTraining : QueryWrapper = QueryWrapper(math.ceil(config["numberOfEachQuery"] * config["trainTestSplit"]))
    origRtreeParamsTraining : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours
    

    origRtreeQueriesTraining.createRangeQueries(origRtree, origRtreeParamsTraining)
    origRtreeQueriesTraining.createSimilarityQueries(origRtree, origRtreeParamsTraining)
    # origRtreeQueriesTraining.createKNNQueries(origRtree, origRtreeParamsTraining)
    origRtreeQueriesTraining.createClusterQueries(origRtree, origRtreeParamsTraining)

    # ---- Create evaluation queries -----
    origRtreeQueriesEvaluation : QueryWrapper = QueryWrapper(math.floor(config["numberOfEachQuery"] - config["numberOfEachQuery"] * config["trainTestSplit"]))
    origRtreeParamsEvaluation : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours

    origRtreeQueriesEvaluation.createRangeQueries(origRtree, origRtreeParamsEvaluation)
    origRtreeQueriesEvaluation.createSimilarityQueries(origRtree, origRtreeParamsEvaluation)
    # origRtreeQueriesEvaluation.createKNNQueries(origRtree, origRtreeParamsEvaluation)
    origRtreeQueriesEvaluation.createClusterQueries(origRtree, origRtreeParamsEvaluation)

    
    compressionRateScores = list()

    #origTrajectoriesSize = sum(list(map(lambda T: len(T.nodes), origTrajectories)))

    ## Main Loop
    #print("Main loop..")
    for cr in tqdm(config["compression_rate"], desc="compression rate"):        
        giveQueryScorings(origRtree, origTrajectories, origRtreeQueriesTraining)
        simpTrajectories = dropNodes(origRtree, origTrajectories, cr)

        simpRtree, simpTrajectories = loadRtree(SIMPLIFIEDDATABASENAME, simpTrajectories)

        compressionRateScores.append({ 'cr' : cr, 'avgf1' : getAverageF1ScoreAll(origRtreeQueriesEvaluation, origRtree, simpRtree), 'simplificationError' : GetSimplificationError(ORIGTrajectories, simpTrajectories), 'simplifiedTrajectories' : copy.deepcopy(simpTrajectories)}) #, GetSimplificationError(origTrajectories, simpTrajectories)
        # While above compression rate
        print(compressionRateScores[-1]['avgf1'])
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




if __name__ == "__main__":
    config = {}
    config["epochs"] = 100                  # Number of epochs to simplify the trajectory database
    config["compression_rate"] = [0.5]      # Compression rate of the trajectory database
    config["DB_size"] = 100                 # Amount of trajectories to load (Potentially irrelevant)
    config["verbose"] = True                # Print progress
    config["trainTestSplit"] = 0.8          # Train/test split
    config["numberOfEachQuery"] = 200      # Number of queries used to simplify database    

    main(config)