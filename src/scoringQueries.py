from QueryWrapper import QueryWrapper
from tqdm import tqdm

def giveQueryScorings(Rtree, trajectories, queryWrapper : QueryWrapper):
    # Extract all queries
    print("Scoring queries..")
    for Query in tqdm(queryWrapper.getQueries()):#[queryWrapper.RangeQueries + queryWrapper.KNNQueries + queryWrapper.SimilarityQueries + queryWrapper.ClusterQueries]:
        # Get result of query
        result = Query.run(Rtree)
        # Distribute points
        Query.distribute(trajectories, result)
    print("Done!")
