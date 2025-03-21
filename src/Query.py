from abc import ABC, abstractmethod
from src.Trajectory import Trajectory

# Define the abstract query class 
class Query(ABC):
    def __init__(self, params):
        self.params = params
        
    # run query which returns array of Trajectory class 
    @abstractmethod
    def run(self, rtree) -> list[Trajectory]:
        pass

    @abstractmethod
    def distribute(self, trajectories: list[Trajectory], matches) -> None:
        pass
    
    def __iter__(self):
        return self