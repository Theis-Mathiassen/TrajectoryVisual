from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
from src.Util import ParamUtil
from src.QueryWrapper import QueryWrapper
from src.scoringQueries import giveQueryScorings
from src.load import build_Rtree, load_Tdrive, loadRtree
from src.dropNodes import dropNodes

import os
import sys
import copy
from tqdm import tqdm

sys.path.append("src/")
#### main
def main(config):
    ## Load Dataset
    load_Tdrive("first_10000_train.csv","first_10000_train_trimmed.csv")
    
    origRtree, origTrajectories = build_Rtree("first_10000_train_trimmed.csv", filename="original_Taxi")
    #simpRtree, simpTrajectories = build_Rtree("first_10000_train_trimmed.csv", filename="simplified_Taxi")
    ## Setup reinforcement learning algorithms (t2vec, etc.)

    ORIGTrajectories = copy.deepcopy(origTrajectories)

    ## Setup data collection environment, that is evaluation after each epoch
    origRtreeQueries : QueryWrapper = QueryWrapper(config["numberOfEachQuery"])
    origRtreeParams : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours
    
    origRtreeQueries.createRangeQueries(origRtree, origRtreeParams)
    # origRtreeQueries.createSimilarityQueries(origRtree, origRtreeParams)
    # origRtreeQueries.createKNNQueries(origRtree, origRtreeParams)
    # origRtreeQueries.createClusterQueries(origRtree, origRtreeParams)
    
    compressionRateScores = list()

    #origTrajectoriesSize = sum(list(map(lambda T: len(T.nodes), origTrajectories)))

    ## Main Loop
    print("Main loop..")
    for cr in tqdm(config["compression_rate"]):        
        giveQueryScorings(origRtree, origTrajectories, origRtreeQueries)
        simpTrajectories = dropNodes(origRtree, origTrajectories, cr)

        simpRtree, simpTrajectories = loadRtree("simplified_Taxi", simpTrajectories)

        compressionRateScores.append((cr, getAverageF1ScoreAll(origRtreeQueries, origRtree, simpRtree), GetSimplificationError(ORIGTrajectories, simpTrajectories))) #, GetSimplificationError(origTrajectories, simpTrajectories)
        # While above compression rate
        
        # Generate and apply queries, giving scorings to points

        # Remove x points with the fewest points

        # Collect evaluation data
            # getAverageF1ScoreAll, GetSimplificationError

    ## Save results
    for res in compressionRateScores:
        print(res)

    ## Plot models

    pass




if __name__ == "__main__":
    config = {}
    config["epochs"] = 100                  # Number of epochs to simplify the trajectory database
    config["compression_rate"] = [0.5]      # Compression rate of the trajectory database
    config["DB_size"] = 100                 # Amount of trajectories to load (Potentially irrelevant)
    config["verbose"] = True                # Print progress
    config["numberOfEachQuery"] = 200      # Number of queries used to simplify database    

    main(config)