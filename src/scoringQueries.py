from src.QueryWrapper import QueryWrapper
from src.clusterQuery import ClusterQuery
from tqdm import tqdm

def giveQueryScorings(Rtree, trajectories, queryWrapper : QueryWrapper):
    # Extract all queries
    #print("Scoring queries..")
    for Query in tqdm(queryWrapper.getQueries(),desc="Scoring queries"):#[queryWrapper.RangeQueries + queryWrapper.KNNQueries + queryWrapper.SimilarityQueries + queryWrapper.ClusterQueries]:

        # Distribute points
        if not isinstance(Query, ClusterQuery):
            # Get result of query
            result = Query.run(Rtree)
            Query.distribute(trajectories, result)
        else:
            Query.distribute(trajectories)
    #print("Done!")
