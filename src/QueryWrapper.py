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
        
    def createRangeQueries(self, rtree, paramUtil : ParamUtil):
        if self.random:
            for query in range(self.numberOfEachQuery):
                self.RangeQueries.append(RangeQuery(paramUtil.rangeParams(rtree)))
        else:
            for trajectory in self.trajectories.values():
                self.RangeQueries.append(RangeQuery(paramUtil.rangeParams(rtree, index=trajectory.id)))
        
    def createKNNQueries(self, rtree, paramUtil : ParamUtil):
        if self.random:
            for query in range(self.numberOfEachQuery):
                self.KNNQueries.append(KnnQuery(paramUtil.knnParams(rtree)))
        else:
            for trajectory in self.trajectories.values():
                self.KNNQueries.append(KnnQuery(paramUtil.knnParams(rtree, index=trajectory.id)))
    
    def createSimilarityQueries(self, rtree, paramUtil : ParamUtil):
        if self.random:
            for query in range(self.numberOfEachQuery):
                self.SimilarityQueries.append(SimilarityQuery(paramUtil.similarityParams(rtree)))
        else:
            for trajectory in self.trajectories.values():
                self.SimilarityQueries.append(SimilarityQuery(paramUtil.similarityParams(rtree, index=trajectory.id)))
        
    def createClusterQueries(self, rtree, paramUtil : ParamUtil):
        #self.ClusterQueries.append(ClusterQuery(paramUtil.clusterParams(rtree)))
        if self.random:
            for query in range(self.numberOfEachQuery):
                self.ClusterQueries.append(ClusterQuery(paramUtil.clusterParams(rtree)))
        else:
            for trajectory in self.trajectories.values():
                self.ClusterQueries.append(ClusterQuery(paramUtil.clusterParams(rtree, index=trajectory.id)))
                    
    def getQueries(self):
        return [*self.RangeQueries, *self.SimilarityQueries, *self.KNNQueries, *self.ClusterQueries]