from src.QueryWrapper import QueryWrapper
from src.clusterQuery import ClusterQuery
from src.rangeQuery import RangeQuery
from src.similarityQuery import SimilarityQuery
from src.knnQuery import KnnQuery
from tqdm import tqdm
import pickle
import random
from src.log import logger

def giveQueryScorings(Rtree, trajectories, queryWrapper : QueryWrapper = None, pickleFiles = None, numberToTrain = None, config = None):
    if queryWrapper is not None and pickleFiles is None:
        # Extract all queries
        for query in tqdm(queryWrapper.getQueries(),desc="Scoring queries"):#[queryWrapper.RangeQueries + queryWrapper.KNNQueries + queryWrapper.SimilarityQueries + queryWrapper.ClusterQueries]:
            logger.info('Gives scores for query %s', type(query))
            # Get result of query
            result = query.run(Rtree, trajectories)
            # Distribute points
            if not isinstance(query, ClusterQuery):
                query.distribute(trajectories, result)
            else:
                query.distribute(trajectories)
    elif pickleFiles is not None: 
        for filename in pickleFiles:
            logger.info('Pickle file already exists with name: %s', filename)
            with open(filename, 'rb') as f:
                hits = pickle.load(f)

                # Set amount of queries to run
                amountToRun = len(hits)
                if numberToTrain is not None:
                    amountToRun = min(amountToRun, numberToTrain)
                    random.shuffle(hits)
                    #hits.shuffle() # Randomize

                for i in tqdm(range(amountToRun), desc="Scoring queries"):
                    query, result = hits[i]

                    setDistributeType(config, query) # Set query distribute type based on config

                    if not isinstance(query, ClusterQuery): # no cluster query for now
                        query.distribute(trajectories, result)
                    else:
                        query.distribute(trajectories)


def setDistributeType(config, query):
    if config is None:
        return
    elif isinstance(query, RangeQuery):
        query.flag = config.range_flag
    elif isinstance(query, SimilarityQuery):
        query.scoringSystem = config.similarity_system

