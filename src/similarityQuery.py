from rtree import index
import os
import numpy as np
from src.Trajectory import Trajectory
from src.Node import Node
from src.Query import Query

class SimilarityQuery(Query):
    trajectory: Trajectory
    t1: float
    t2: float
    delta: float
    streak: int
    
    def __init__(self, params):
        self.trajectory = params["origin"]
        self.t1 = params["t1"]
        self.t2 = params["t2"]
        self.delta = params["delta"]
        self.scoringSystem = "a"    # Between "c", "a", "c+f", "m"
                                    # C -> Closest
                                    # A -> All
                                    # c+f -> Closest + Farthest
                                    # m -> moving away for a longer period than streak
        self.streak = 2


    def run(self, rtree):

        # For each node in the trajectory
        trajectory_hits = {}
        for node in self.trajectory.nodes:
            x = node.x 
            y = node.y 
            t = node.t

            # If fits time range
            if (self.t1 <= t and t <= self.t2):
                point1 = np.array((node.x, node.y))

                # Get all points possibly within range with range query

                maybe_hits = list(rtree.intersection((x - self.delta, y - self.delta, t, x + self.delta, y + self.delta, t), objects=True))

                for maybe_hit in maybe_hits:
                    (x_idx, y_idx, t_idx, _, _, _) = maybe_hit.bbox
                    trajectory_id, node_id = maybe_hit.object

                    # If fit time, and not part of the queried trajectory
                    if t == t_idx and trajectory_id != self.trajectory.id:
                        point2 = np.array((x_idx, y_idx))

                        # calculating Euclidean distance to find if actual hit
                        dist = np.linalg.norm(point1 - point2)
                        if dist <= self.delta:
                            node = Node(node_id, x_idx, y_idx, t_idx)

                            # Get list of nodes by trajectories
                            if trajectory_id not in trajectory_hits:
                                trajectory_hits[trajectory_id] = []

                            trajectory_hits[trajectory_id].append(node)

        # Convert to corrct format
        trajectories = [Trajectory(trajectory_id, nodes) for trajectory_id, nodes in trajectory_hits.items()]
        
        return trajectories
    
    def distribute(self, trajectories, matches):
        nodesIdsReward = [] #In the form (trajectory Id, node Id)

        # Get node ids to reward
        if self.scoringSystem == "c": #Reward closest
            trajectoriesWithSortedNodes = self.getDistanceSortedTrajectory(matches)
            for (trajectory_id, node_ids) in trajectoriesWithSortedNodes:
                nodesIdsReward.append((trajectory_id, node_ids[0]))

        elif self.scoringSystem == "a": #Reward all
            for trajectory in matches:
                for node in trajectory.nodes:
                    nodesIdsReward.append((trajectory.id, node.id))

        elif self.scoringSystem == "c+f": #Reward closest and furthest
            trajectoriesWithSortedNodes = self.getDistanceSortedTrajectory(matches)
            for (trajectory_id, node_ids) in trajectoriesWithSortedNodes:
                nodesIdsReward.append((trajectory_id, node_ids[0]))
                if len(node_ids) != 1: # Get furthest only if there is more than 1 node
                    nodesIdsReward.append((trajectory_id, node_ids[-1]))

        elif self.scoringSystem == "m": #Reward those moving away, with a streak of more than self.streak
            trajectoriesWithMovingAwayScores = self.getMovingAwayScore(matches)

            for (trajectoryId, movingAwayScores) in trajectoriesWithMovingAwayScores:
                for (nodeId, streakScore) in movingAwayScores:
                    if streakScore > self.streak:
                        nodesIdsReward.append((trajectoryId, nodeId))

        # Sort for faster scores
        nodesIdsRewardSorted = sorted(nodesIdsReward, key = lambda x: x[0])

        relevantTrajectories = [x for x in trajectories if x.id in [x[0] for x in nodesIdsRewardSorted]]
        relevantTrajectoriesSorted = sorted(relevantTrajectories, key = lambda x: x.id)

        # Hand out scores
        relevantTrajectoriesIndex = 0

        for (trajectory_id, node_id) in nodesIdsRewardSorted:
            # Find correct index
            while(relevantTrajectoriesSorted[relevantTrajectoriesIndex].id != trajectory_id):
                relevantTrajectoriesIndex += 1
            
            # Find node index
            nodeIndex = [x.id for x in relevantTrajectoriesSorted[relevantTrajectoriesIndex].nodes].index(node_id)  # We have to convert to ids so we don't look through pointers

            relevantTrajectoriesSorted[relevantTrajectoriesIndex].nodes[nodeIndex].score += 1
            


    # For each trajectory get a sorted list of node ids based on distance to origin (Ascending)
    def getDistanceSortedTrajectory(self, matches, getWithDistance = False):
        
        trajectoriesWithSortedNodes = [] #In the form (trajectory id, [node id, node id])

        # get origin nodes in time window
        originNodesInTimeWindow = [x for x in self.trajectory.nodes if (self.t1 <= x.t and x.t <= self.t2)]


        for matchTrajectory in matches:
            nodes = {}

            for matchNode in matchTrajectory.nodes:
                minDist = self.delta #Set min to delta, till min is found

                point1 = np.array((matchNode.x, matchNode.y))

                for originNode in originNodesInTimeWindow:
                    point2 = np.array((originNode.x, originNode.y))

                    # calculating Euclidean distance
                    dist = np.linalg.norm(point1 - point2)

                    minDist = min(dist, minDist)
                
                # Save min distance
                nodes[matchNode.id] = minDist
            
            sortedList = []
            if getWithDistance:
                sortedList = [(x[0], x[1]) for x in nodes.items()]
            else:
                #Sort by distance
                sortedDict = sorted(nodes.items(), key = lambda x: x[1])
                sortedList = [x[0] for x in sortedDict]

            trajectoriesWithSortedNodes.append((matchTrajectory.id, sortedList))

        return trajectoriesWithSortedNodes
   
    # Gets moving away score for a single trajectory
    def getMovingAwayScore(self, matches):

        trajectoriesWithSortedNodes = self.getDistanceSortedTrajectory(matches, getWithDistance=True)

        trajectoriesMovingAwayScores = []

        for (trajectoryId, sortedList) in trajectoriesWithSortedNodes:
            
            movingAwayScores = [] # Of the form (node id, amount of steps where it has been moving id)

            latest = -1
            streak = 0 # Describes the amount of times we have been moving away

            for (node_id, score) in sortedList:
                if score > latest:
                    streak += 1
                else :
                    streak = 0
                movingAwayScores.append((node_id, streak))
            
                latest = score

            trajectoriesMovingAwayScores.append((trajectoryId, movingAwayScores))
        
        return trajectoriesMovingAwayScores


if __name__== "__main__":
    import os
    import sys
    from Util import ParamUtil


    currentPath = os.getcwd()

    print("CurrentPath: ", currentPath)

    sys.path.append(currentPath + "/")

    from load import build_Rtree

    rtree, trajectories = build_Rtree("trimmed_small_train.csv", "simplified_Tdrive")

    paramUtil = ParamUtil(rtree, trajectories, delta = 500000000)

    params = paramUtil.similarityParams(rtree, delta= 5000000)

    myQuery = SimilarityQuery(params)

    results = myQuery.run(rtree)

    print("Results len: ", len(results))

    print("Origin id: ", myQuery.trajectory.id)
    for x in results:
        print("Result id :", x.id)



    print("---- Starting distribute ---- \n")
    myQuery.distribute(trajectories, results)

    for trajectory in trajectories:
        if trajectory.id in [x.id for x in results]:
            print("\n----- Trajetory id: " + str(trajectory.id) +" -----")
            for node in trajectory.nodes:
                print("Node id: ", node.id, "Node Score: ", node.score)






