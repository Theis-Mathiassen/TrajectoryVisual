from Query import Query

class QueryWrapper:
    def __init__(self, numberOfEachQuery):
        self.numberOfEachQuery = numberOfEachQuery
        self.RangeQueries = list[Query]
        self.KNNQueries = list[Query]
        self.SimilarityQueries = list[Query]
        self.ClusterQueries = list[Query]
        
    def createRangeQueries(self, rtree, paramUtil):
        for query in range(self.numberOfEachQuery):
            self.RangeQueries.append(Query(paramUtil.params(rtree)))
        
    def createKNNQueries(self, rtree, paramUtil):
        for query in range(self.numberOfEachQuery):
            self.KNNQueries.append(Query(paramUtil.params(rtree)))
    
    def createSimilarityQueries(self, rtree, paramUtil):
        for query in range(self.numberOfEachQuery):
            self.SimilarityQueries.append(Query(paramUtil.params(rtree)))
    
    def createClusterQueries(self, rtree, paramUtil):
        for query in range(self.numberOfEachQuery):
            self.ClusterQueries.append(Query(paramUtil.params(rtree)))