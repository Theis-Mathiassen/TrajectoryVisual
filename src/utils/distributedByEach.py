import os
import sys

# Find the absolute path of the project directory
absolute_path = os.path.dirname(__file__)

# Define the relative paths of directories to import from
relative_path_src = "../.."
relative_path_test = "../../src"

# Create the full path for these directories
full_path_src = os.path.join(absolute_path, relative_path_src)
full_path_test = os.path.join(absolute_path, relative_path_test)

# Append them to the path variable of the system
sys.path.append(full_path_src)
sys.path.append(full_path_test)


from src.QueryWrapper import QueryWrapper
from src.clusterQuery import ClusterQuery
from src.rangeQuery import RangeQuery
from src.similarityQuery import SimilarityQuery
from src.knnQuery import KnnQuery
from tqdm import tqdm
import pickle
import random
from src.log import logger
import matplotlib.pyplot as plt
import numpy as np


from src.load import get_Tdrive

PICKLE_HITS = ['RangeQueryHits.pkl', 'KnnQueryHits.pkl', 'SimilarityQueryHits.pkl'] 

RANGE_CONFIGS = [1, 2, 3, 4]  
SIMILARITY_CONFIGS = ["c", "a", "c+f", "m"]

DATABASENAME = 'original_Taxi'


MAX_QUERIES = 1000
INTERVALS = 20



# Load trajectories
origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)



def getTotalScore(trajectories):
    totalScore = 0
    for traj in trajectories.values():
        for node in traj.nodes:
            totalScore += node.score
    return totalScore

def getAllScores (trajectories):
    scores = []
    for traj in trajectories.values():
        for node in traj.nodes:
            if node.score != 0:
                scores.append(node.score)
    return scores


resultTotal = {}
resultScores= {}


for filename in PICKLE_HITS:
    with open(filename, 'rb') as f:
        hits = pickle.load(f)

        
        maxQueries = min(MAX_QUERIES, len(hits))

        queryType = ""
        configurations = ["standard"]
        query, _ = hits[0]
        if isinstance(query, RangeQuery):
            queryType = "range"
            configurations = RANGE_CONFIGS
        elif isinstance(query, SimilarityQuery):
            queryType = "similarity"
            configurations = SIMILARITY_CONFIGS
        elif isinstance(query, KnnQuery):
            queryType = "knn"
        elif isinstance(query, ClusterQuery):
            queryType = "cluster"


        for distributeType in configurations:
            resultTotal["{}-{}".format(queryType, distributeType)] = []

            for i in tqdm(range(maxQueries), desc=f"Scoring {queryType} queries with {distributeType}"):
                # Change query distribute type
                query, result = hits[i]
                if distributeType == "range":
                    query.flag = distributeType
                elif distributeType == "similarity":
                    query.scoringSystem = distributeType

                # Distribute points
                if not isinstance(query, ClusterQuery): # no cluster query for now
                    query.distribute(origTrajectories, result)
                else:
                    query.distribute(origTrajectories)

                # Get total score and add to list
                if i % INTERVALS == 0:
                    resultTotal["{}-{}".format(queryType, distributeType)].append(getTotalScore(origTrajectories))


            resultScores["{}-{}".format(queryType, distributeType)] = getAllScores(origTrajectories)    
            

# Moving to plotting
# Plot total scores over amount of queries
amount = np.arrange(0, MAX_QUERIES, INTERVALS)
for queryType in resultTotal.keys():
    plt.plot(amount, resultTotal[queryType], label=queryType)


plt.xlabel("Amount Of Queries")
plt.legend("Total Score")
plt.savefig('totalScores.jpg')
plt.show()

# Plot score histograms
for queryType in resultScores.keys():
    plt.hist(resultScores[queryType], bins=100)

    plt.legend()

    plt.savefig(queryType + '_scores.jpg')

    plt.xlabel("Score")
    plt.ylabel("Frequency")

    plt.show()

