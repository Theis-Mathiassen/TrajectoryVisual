from src.QueryWrapper import QueryWrapper
from src.clusterQuery import ClusterQuery
from tqdm import tqdm
import pickle
import random
from src.log import logger

def giveQueryScorings(Rtree, trajectories, numberToTrain, queryWrapper : QueryWrapper = None, pickleFiles = None):
    if queryWrapper is not None and pickleFiles is None:
        # Extract all queries
        for query in tqdm(queryWrapper.getQueries(),desc="Scoring queries"):#[queryWrapper.RangeQueries + queryWrapper.KNNQueries + queryWrapper.SimilarityQueries + queryWrapper.ClusterQueries]:
            logger.info('Gives scores for query %s', type(query))
            # Get result of query
            result = query.run(Rtree)
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
                # listOfQueriesIdx = random.sample(range(0, len(hits)), numberToTrain // len(pickleFiles))
                for idx in tqdm(hits, desc="Scoring queries"):
                    query, result = hits[idx]
                    if not isinstance(query, ClusterQuery): # no cluster query for now
                        query.distribute(trajectories, result)
                    else:
                        query.distribute(trajectories)