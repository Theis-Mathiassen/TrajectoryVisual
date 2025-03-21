from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
from src.Util import ParamUtil
from src.QueryWrapper import QueryWrapper
from src.scoringQueries import giveQueryScorings
from load import build_Rtree, load_Tdrive, loadRtree
from src.dropNodes import dropNodes

import os
from tqdm import tqdm

#### main
def main(config):
    ## Load Dataset
    #load_Tdrive("small_train.csv","small_train_trimmed.csv")
    
    origRtree, origTrajectories = build_Rtree("first_100000_train_trimmed.csv", filename="original_Taxi")
    simpRtree, simpTrajectories = build_Rtree("first_100000_train_trimmed.csv", filename="simplified_Taxi")
    ## Setup reinforcement learning algorithms (t2vec, etc.)


    ## Setup data collection environment, that is evaluation after each epoch
    origRtreeQueries : QueryWrapper = QueryWrapper(config["numberOfEachQuery"])
    origRtreeParams : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours
    
    origRtreeQueries.createRangeQueries(origRtree, origRtreeParams)
    origRtreeQueries.createSimilarityQueries(origRtree, origRtreeParams)
    
    compressionRateScores = list()

    origTrajectoriesSize = sum(list(map(lambda T: len(T.nodes), origTrajectories)))

    ## Main Loop
    print("Main loop..")
    for cr in tqdm(config["compression_rate"]):
        compressedTrajectoriesSize = cr * origTrajectoriesSize
        while(sum(list(map(lambda T: len(T.nodes), simpTrajectories))) > (compressedTrajectoriesSize)):
            giveQueryScorings(simpRtree, simpTrajectories, origRtreeQueries)
            dropNodes(simpRtree, simpTrajectories, cr)

        compressionRateScores.append((cr, getAverageF1ScoreAll(origRtreeQueries, origRtree, simpRtree), GetSimplificationError(origTrajectories, simpTrajectories)))

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
    config["numberOfEachQuery"] = 100      # Number of queries used to simplify database    

    main(config)