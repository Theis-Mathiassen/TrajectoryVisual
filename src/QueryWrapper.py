from src.Query import Query
from src.Util import ParamUtil
from src.rangeQuery import RangeQuery
from src.clusterQuery import ClusterQuery
from src.knnQuery import KnnQuery
from src.similarityQuery import SimilarityQuery

class QueryWrapper:
    def __init__(self, numberOfEachQuery):
        self.numberOfEachQuery = numberOfEachQuery
        self.RangeQueries = []#list[RangeQuery]
        self.KNNQueries = []#list[KnnQuery]
        self.SimilarityQueries = []#list[SimilarityQuery]
        self.ClusterQueries = [] #list[ClusterQuery]
        
    def createRangeQueries(self, rtree, paramUtil : ParamUtil):
        for query in range(self.numberOfEachQuery):
            self.RangeQueries.append(RangeQuery(paramUtil.rangeParams(rtree)))
        
    def createKNNQueries(self, rtree, paramUtil : ParamUtil):
        for query in range(self.numberOfEachQuery):
            self.KNNQueries.append(KnnQuery(paramUtil.knnParams(rtree)))
    
    def createSimilarityQueries(self, rtree, paramUtil : ParamUtil):
        for query in range(self.numberOfEachQuery):
            self.SimilarityQueries.append(SimilarityQuery(paramUtil.similarityParams(rtree)))
    
    def createClusterQueries(self, rtree, paramUtil : ParamUtil):
        self.ClusterQueries.append(ClusterQuery(paramUtil.clusterParams(rtree)))
            
    def getQueries(self):
        return [*self.RangeQueries, *self.SimilarityQueries, *self.KNNQueries, *self.ClusterQueries]