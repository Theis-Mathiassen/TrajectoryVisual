from src.QueryWrapper import QueryWrapper
from src.clusterQuery import ClusterQuery
from tqdm import tqdm
from src.log import logger


def giveQueryScorings(Rtree, trajectories, queryWrapper : QueryWrapper = None, pickleFiles = None):
    if queryWrapper is not None and pickleFiles is None:
        # Extract all queries
        #print("Scoring queries..")
        for Query in tqdm(queryWrapper.getQueries(),desc="Scoring queries"):#[queryWrapper.RangeQueries + queryWrapper.KNNQueries + queryWrapper.SimilarityQueries + queryWrapper.ClusterQueries]:
            logger.info('Gives scores for query %s', type(Query))
            # Get result of query
            result = Query.run(Rtree)
            # Distribute points
            if not isinstance(Query, ClusterQuery):
                Query.distribute(trajectories, result)
            else:
                Query.distribute(trajectories)
    elif pickleFiles is not None: 
        for filename in pickleFiles:
            logger.info('Pickle file already exists with name: %s', filename)
            with open(filename, 'rb') as f:
                for q, r in f:
                    if not isinstance(q, ClusterQuery):
                        Query.distribute(trajectories, r)
                    else:
                        Query.distribute(trajectories)

    #print("Done!")
