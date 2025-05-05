from src.Query import Query
from src.Util import ParamUtil
from src.rangeQuery import RangeQuery
from src.clusterQuery import ClusterQuery
from src.knnQuery import KnnQuery
from src.similarityQuery import SimilarityQuery

class QueryWrapper:
    def __init__(self, numberOfEachQuery, random = True, trajectories = None):
        self.numberOfEachQuery = numberOfEachQuery
        self.RangeQueries = []#list[RangeQuery]
        self.KNNQueries = []#list[KnnQuery]
        self.SimilarityQueries = []#list[SimilarityQuery]
        self.ClusterQueries = [] #list[ClusterQuery]
        self.random = random
        self.trajectories = trajectories
        
    def createRangeQueries(self, rtree, paramUtil : ParamUtil, flag: int = 1):
        if self.random:
            for query in range(self.numberOfEachQuery):
                params = paramUtil.rangeParams(rtree)
                params["flag"] = flag
                self.RangeQueries.append(RangeQuery(params))
        else:
            for trajectory in self.trajectories.values():
                params = paramUtil.rangeParams(rtree, index=trajectory.id)
                params["flag"] = flag
                self.RangeQueries.append(RangeQuery(params))
        
    def createKNNQueries(self, rtree, paramUtil : ParamUtil, distance_method: int = 1):
        if self.random:
            for query in range(self.numberOfEachQuery):
                params = paramUtil.knnParams(rtree)
                params["distanceMethod"] = distance_method
                self.KNNQueries.append(KnnQuery(params))
        else:
            for trajectory in self.trajectories.values():
                params = paramUtil.knnParams(rtree, index=trajectory.id)
                params["distanceMethod"] = distance_method
                self.KNNQueries.append(KnnQuery(params))
    
    def createSimilarityQueries(self, rtree, paramUtil : ParamUtil, scoring_system: str = "c"):
        if self.random:
            for query in range(self.numberOfEachQuery):
                params = paramUtil.similarityParams(rtree)
                params["scoringSystem"] = scoring_system
                self.SimilarityQueries.append(SimilarityQuery(params))
        else:
            for trajectory in self.trajectories.values():
                params = paramUtil.similarityParams(rtree, index=trajectory.id)
                params["scoringSystem"] = scoring_system
                self.SimilarityQueries.append(SimilarityQuery(params))
        
    def createClusterQueries(self, rtree, paramUtil : ParamUtil):
        self.ClusterQueries.append(ClusterQuery(paramUtil.clusterParams(rtree)))
            
    def getQueries(self):
        return [*self.RangeQueries, *self.SimilarityQueries, *self.KNNQueries, *self.ClusterQueries]