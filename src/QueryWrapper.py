from src.Query import Query
from src.Util import ParamUtil
from src.rangeQuery import RangeQuery
from src.clusterQuery import ClusterQuery
from src.knnQuery import KnnQuery
from src.similarityQuery import SimilarityQuery

import numpy as np
import random
from rtree import index

class QueryWrapper:
    def __init__(self, numberOfEachQuery, random = True, trajectories = None, useGaussian = False, avgCoordinateValues = None, rtree: index.Index = None, sigma = 500):
        self.numberOfEachQuery = numberOfEachQuery
        self.RangeQueries = []#list[RangeQuery]
        self.KNNQueries = []#list[KnnQuery]
        self.SimilarityQueries = []#list[SimilarityQuery]
        self.ClusterQueries = [] #list[ClusterQuery]
        self.random = random
        self.trajectories = trajectories
        self.useGaussian = useGaussian
        self.avgCoordinateValues = avgCoordinateValues
        self.rtree = rtree
        self.sigma = sigma
        self.bounds = rtree.bounds
        
    def createRangeQueries(self, rtree, paramUtil : ParamUtil, flag: int = 1):
        if self.random:
            for query in range(self.numberOfEachQuery):
                params = paramUtil.rangeParams(rtree)
                params["flag"] = flag
                self.RangeQueries.append(RangeQuery(params))
        elif self.useGaussian:
            for query in range(self.numberOfEachQuery):
                randomPoint = (np.random.normal(self.avgCoordinateValues, self.sigma, size=(1, 3)))[0]
                randomPoint[0] = np.clip(randomPoint[0], self.bounds[0], self.bounds[3])
                randomPoint[1] = np.clip(randomPoint[1], self.bounds[1], self.bounds[4])
                randomPoint[2] = np.clip(randomPoint[2], self.bounds[2], self.bounds[5])
                params = paramUtil.gaussianRangeParams(randomPoint)
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
        elif self.useGaussian:
            for query in range(self.numberOfEachQuery):
                randomPoint = (np.random.normal(self.avgCoordinateValues, self.sigma, size=(1, 3)))[0]
                (trajectory_id, node_id) = self._getNearestNode(randomPoint)          
                params = paramUtil.knnParams(rtree, index=trajectory_id, nodeIndex=node_id)
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
        elif self.useGaussian:
            for query in range(self.numberOfEachQuery):
                randomPoint = (np.random.normal(self.avgCoordinateValues, self.sigma, size=(1, 3)))[0]
                (trajectory_id, node_id) = self._getNearestNode(randomPoint)          
                params = paramUtil.similarityParams(rtree, index=trajectory_id, nodeIndex=node_id)
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
    
    def _getNearestNode(self, point):
        xcord = np.clip(point[0], self.bounds[0], self.bounds[3])
        ycord = np.clip(point[1], self.bounds[1], self.bounds[4])
        t = np.clip(point[2], self.bounds[2], self.bounds[5])
        nearestList = list(self.rtree.nearest((xcord, ycord, t, xcord, ycord, t), 1, objects="raw")) # If multiple nodes are equal distance they are all returned, despite only getting top 1

        matches = len(nearestList)
        if matches == 0:
            raise Exception("No nearest node found")
        elif matches == 1:
            return nearestList[0]
        
        # The list is returned ordered by node index, so we should shuffle for fairness
        return random.choice(nearestList)

        