from abc import ABC, abstractmethod
from src.Trajectory import Trajectory

# Define the abstract Filter class 
class Filter(ABC):
    def __init__(self, params):
        self.params = params
        
    # filterTrajectories returns the filtered trajectories
    @abstractmethod
    def filterTrajectories(self, trajectories: list[Trajectory]) -> None:
        pass 
        
    
    def __iter__(self):
        return self