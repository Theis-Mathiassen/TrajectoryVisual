import itertools
from dataclasses import dataclass
from typing import Union, List

@dataclass
class Configuration:
    epochs: int
    compression_rate: Union[float, List[float]]  # Can be either a single float or a list of floats
    DB_size: int
    trainTestSplit: float
    numberOfEachQuery: int
    QueriesPerTrajectory: float
    verbose: bool = True  


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
            print(f"Error creating Configuration object for tuple {combination_tuple}: {e}")
            print(f"Check if the number of elements in the tuple matches the "
                    f"expected arguments for {Configuration.__name__}.__init__")
            continue

    return all_config_objects
