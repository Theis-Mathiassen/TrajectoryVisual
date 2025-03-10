from QueryWrapper import QueryWrapper

def giveQueryScorings(Rtree, trajectories, queryWrapper : QueryWrapper):
        # Extract all queries
        for Query in queryWrapper.getQueries():#[queryWrapper.RangeQueries + queryWrapper.KNNQueries + queryWrapper.SimilarityQueries + queryWrapper.ClusterQueries]:
            # Get result of query
            result = Query.run(Rtree)
            # Distribute points
            Query.distribute(trajectories, result)
