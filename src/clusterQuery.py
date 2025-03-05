
from Query import Query


#This is a temporary holder. Used for handling cases in evaluation such that it is not evaluated falsely
class ClusterQuery(Query):

    def __init__(self, params):
        pass

    def run(self, rtree):
        return []