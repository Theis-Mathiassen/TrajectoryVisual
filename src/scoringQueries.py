from src.QueryWrapper import QueryWrapper
from src.clusterQuery import ClusterQuery
from tqdm import tqdm
import pickle
import random

def giveQueryScorings(Rtree, trajectories, numberToTrain, queryWrapper : QueryWrapper = None, pickleFiles = None):
    if queryWrapper is not None and pickleFiles is None:
        # Extract all queries
        #print("Scoring queries..")
        for Query in tqdm(queryWrapper.getQueries(),desc="Scoring queries"):#[queryWrapper.RangeQueries + queryWrapper.KNNQueries + queryWrapper.SimilarityQueries + queryWrapper.ClusterQueries]:
            # Get result of query
            result = Query.run(Rtree)
            # Distribute points
            if not isinstance(Query, ClusterQuery):
                Query.distribute(trajectories, result)
            else:
                Query.distribute(trajectories)
    elif pickleFiles is not None: 
        for filename in pickleFiles:
            with open(filename, 'rb') as f:
                hits = pickle.load(f)
                listOfQueriesIdx = random.sample(range(0, len(hits)), numberToTrain // len(pickleFiles))
                for idx in tqdm(listOfQueriesIdx, desc="Scoring queries"):
                    Query, r = hits[idx]
                    if not isinstance(Query, ClusterQuery):
                        Query.distribute(trajectories, r)
                    else:
                        Query.distribute(trajectories)
    #print("Done!")
