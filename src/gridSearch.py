import itertools
from dataclasses import dataclass
from typing import Union, List
import argparse
from src.log import logger

@dataclass
class Configuration:
    compression_rate: Union[float, List[float]]  # Can be either a single float or a list of floats
    DB_size: int
    trainTestSplit: float
    numberOfEachQuery: int
    QueriesPerTrajectory: float
    verbose: bool = True  
    # Query method parameters
    knn_method: int = 1  # Only 1 is implemented (spatio temporal linear combine distance)
    range_flag: int = 1  # 1-4 for different distribute methods
    similarity_system: str = "c"  # "c", "a", "c+f", "m"
    weights: dict = dict()

def parse_args():
    parser = argparse.ArgumentParser(description='Run trajectory simplification with specified query methods')
    parser.add_argument('--knn', type=int, default=1, choices=[1], 
                      help='KNN distance method (only 1 is implemented)')
    parser.add_argument('--range', type=int, default=1, choices=[1,2,3,4],
                      help='Range query distribute flag (1: winner_takes_all, 2: shared_equally(True), 3: shared_equally(False), 4: gradient_points)')
    parser.add_argument('--similarity', type=str, default="c", choices=["c", "a", "c+f", "m"],
                      help='Similarity query scoring system (c: closest, a: all, c+f: closest+furthest, m: moving away)')
    return parser.parse_args()

def createConfigs(*configs):
    """
    Generates Configuration objects for all combinations of config values.

    Assumes the order of *configs matches the __init__ or dataclass
    field order of the Configuration class.
    """
    all_combinations_tuples = itertools.product(*configs)
    all_config_objects = []

    for combination_tuple in all_combinations_tuples:
        try:
            # Create an instance of Configuration, unpacking the tuple elements
            # as positional arguments to __init__
            config_object = Configuration(*combination_tuple)
            all_config_objects.append(config_object)
        except TypeError as e:
            logger.error(f"Error creating Configuration object for tuple {combination_tuple}: {e}")
            logger.error(f"Check if the number of elements in the tuple matches the "
                    f"expected arguments for {Configuration.__name__}.__init__")
            continue

    return all_config_objects

if __name__ == "__main__":
    print(createConfigs([0.5], [100], [0.8], [100], [0.005, 0.01, 0.02], [True]))
