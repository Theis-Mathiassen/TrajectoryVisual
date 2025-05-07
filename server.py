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
origRtree = None
origTrajectories = None

dictQueryResults = {}



def prepare():
    global origRtree, origTrajectories, jobs_backlog, jobs_active, ready, dictQueryResults


    allCombinations = [(1,1,'a')]


    logger.info("---------------------------    Main Start    ---------------------------")
    
    
    config = {}
    config["epochs"] = 100                  # Number of epochs to simplify the trajectory database
    config["compression_rate"] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]      # Compression rate of the trajectory database
    config["DB_size"] = 100                 # Amount of trajectories to load (Potentially irrelevant)
    config["verbose"] = True                # Print progress
    config["trainTestSplit"] = 0.8          # Train/test split
    config["numberOfEachQuery"] = 5     # Number of queries used to simplify database    
    config["QueriesPerTrajectory"] = 1   # Number of queries per trajectory, in percentage. Overrides numberOfEachQuery if not none


    logger.info('Starting get_Tdrive.')
    origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)
    logger.info('Completed get_Tdrive.')


    

    for (KNN_distanceMethod, RangeQuery_flag, SimilarityQuery_scoringSystem) in allCombinations:
        for filename in PICKLE_HITS:
            logger.info('Pickle file already exists with name: %s', filename)
            with open(filename, 'rb') as f:
                hits = pickle.load(f)
                #for q, r in f: -> for q, r in hits:
                for q, r in tqdm(hits):
                    jobs_backlog.append([KNN_distanceMethod, RangeQuery_flag, SimilarityQuery_scoringSystem, q, r])
                    # q.distribute(origTrajectories, r)
    
    #jobs_backlog = origRtreeQueriesTraining.getQueries()
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
        #return jobs_backlog.pop(0).to_dict(), 200
    
        # Pop the list representing the job from the backlog
        job_details = jobs_backlog.pop(0)

        # Extract the RangeQuery object (assuming it's at index 3)
        range_query_object = job_details[3]

        # Create a dictionary to return, including the serializable RangeQuery data
        # You should also include other relevant details from job_details if needed in the response
        return_data = {
            "status": "success", # Or some other status indicator
            "method": job_details[0], # KNN_distanceMethod
            "flag": job_details[1], # RangeQuery_flag
            "scoring_system": job_details[2], # SimilarityQuery_scoringSystem
            "range_query": range_query_object.to_dict(), # Convert RangeQuery to dict
            "additional_data_r": job_details[4] # Assuming r is at index 4, include if serializable
        }
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