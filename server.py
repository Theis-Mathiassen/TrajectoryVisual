from flask import Flask
from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
from src.Util import ParamUtil
from src.QueryWrapper import QueryWrapper
from src.scoringQueries import giveQueryScorings
from src.load import build_Rtree, load_Tdrive, loadRtree, load_Tdrive_Rtree, get_Tdrive
from src.dropNodes import dropNodes
from src.clusterQuery import ClusterQuery
#import logging
from src.log import logger, ERROR_LOG_FILENAME
import os
import sys
import copy
import pickle
import math
import argparse
from tqdm import tqdm
import pandas as pd 
import logging
import traceback # traceback for information on python stack traces

sys.path.append("src/")

CSVNAME = 'first_10000_train'
DATABASENAME = 'original_Taxi'
SIMPLIFIEDDATABASENAME = 'simplified_Taxi'
LOG_FILENAME = 'script_error_log.log' # Define a log file name
PICKLE_HITS = ['RangeQueryHits.pkl']


ready = False
jobs_backlog = []
jobs_active = []

dictQueryResults = {}



def prepare():

    logger.info("---------------------------    Main Start    ---------------------------")
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run trajectory queries with specified query type')
    parser.add_argument('query_type', type=str, help='Query type to run (range, similarity, knn, or cluster)')
    args = parser.parse_args()
    
    config = {}
    config["epochs"] = 100                  # Number of epochs to simplify the trajectory database
    config["compression_rate"] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]      # Compression rate of the trajectory database
    config["DB_size"] = 100                 # Amount of trajectories to load (Potentially irrelevant)
    config["verbose"] = True                # Print progress
    config["trainTestSplit"] = 0.8          # Train/test split
    config["numberOfEachQuery"] = 5     # Number of queries used to simplify database    
    config["QueriesPerTrajectory"] = 1   # Number of queries per trajectory, in percentage. Overrides numberOfEachQuery if not none
    config["query_type"] = args.query_type  # Query type from command line args


    logger.info('Starting get_Tdrive.')
    origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)
    logger.info('Completed get_Tdrive.')
    """     
    logger.info('Copying trajectories.')
    ORIGTrajectories = {
        tid : copy.deepcopy(traj)
        for tid, traj, in tqdm(origTrajectories.items(), desc = "Copying trajectories")
    } """

    ## Setup data collection environment, that is evaluation after each epoch

    # ---- Set number of queries to be created ----
    if config["QueriesPerTrajectory"] != None : config["numberOfEachQuery"] = math.floor(config["QueriesPerTrajectory"] * len(origTrajectories.values()))


    # ---- Create training queries -----
    logger.info('Creating training queries.')
    origRtreeQueriesTraining : QueryWrapper = QueryWrapper(config["numberOfEachQuery"], random=False, trajectories=origTrajectories)
    origRtreeParamsTraining : ParamUtil = ParamUtil(origRtree, origTrajectories, delta=10800) # Temporal window for T-Drive is 3 hours
    
    # Create the specified query type based on command line argument
    query_type = config["query_type"].lower()
    print(f"Creating {query_type} queries...")
    
    if query_type == "range":
        logger.info('Creating range queries.')
        origRtreeQueriesTraining.createRangeQueries(origRtree, origRtreeParamsTraining)
    elif query_type == "similarity":
        logger.info('Creating similarity queries.')
        origRtreeQueriesTraining.createSimilarityQueries(origRtree, origRtreeParamsTraining)
    elif query_type == "knn":
        logger.info('Creating KNN queries.')
        origRtreeQueriesTraining.createKNNQueries(origRtree, origRtreeParamsTraining)
    elif query_type == "cluster":
        logger.info('Creating cluster queries.')
        origRtreeQueriesTraining.createClusterQueries(origRtree, origRtreeParamsTraining)
    else:
        print(f"Unknown query type: {query_type}")
        print("Available query types: range, similarity, knn, cluster")
        sys.exit(1)

    
    jobs_backlog = origRtreeQueriesTraining.getQueries()
    ready = True


class MyFlaskApp(Flask):
  def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
    if not self.debug or os.getenv('WERKZEUG_RUN_MAIN') == 'true':
      with self.app_context():
        prepare()
    super(MyFlaskApp, self).run(host=host, port=port, debug=debug, load_dotenv=load_dotenv, **options)


app = MyFlaskApp(__name__)



@app.route("/job", methods=["GET"])
def get_job():
    print(f'Length of backlog: {len(jobs_backlog)}')
    if not ready:
        return None, 503
    if len(jobs_backlog) > 0:
        jobs_active.append(jobs_backlog[0])
        return jobs_backlog.pop(0), 200
    else:
        return None, 204
@app.route("/result", methods=["POST"])
def post_result(result):
    
    (q, r) = result
    if str(q) not in dictQueryResults:
        dictQueryResults[str(q)] = []
    dictQueryResults[str(q)].append((q,r))

    if len(jobs_backlog) <= 0 and len(jobs_active) <= 0:
        save()


    return "OK", 200



def save ():
    logger.info('Saving results to pickle files.')
    for queryType in dictQueryResults.keys():    
        try:
            with open(os.path.join(os.getcwd(), str(queryType) + 'Hits.pkl'), 'wb') as file:
                pickle.dump(dictQueryResults[queryType], file)
                file.close()
        except Exception as e:
            print(f"err saving results: {e}")
            logger.error("problems when saving results to pickle file.")
            logger.error(traceback.format_exc())


app.run()