

def giveQueryScorings(Rtree, trajectories, queryWrapper):
        # Extract all queries
        for Query in [queryWrapper.RangeQueries + queryWrapper.KNNQueries + queryWrapper.SimilarityQueries + queryWrapper.ClusterQueries]:
            # Get result of query
            result = Query.run(Rtree, trajectories)
            # Distribute points
            Query.distribute(result, trajectories)
