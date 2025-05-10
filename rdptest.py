import os
import pickle

path = os.getcwd()
    
with open(os.path.join(path, 'rdpTrajectories1_5.pkl'), 'rb') as file:
    trajectoriesWithRdpMask = pickle.load(file)

totalNodes = 0
totalUnmaskedNodes = 0

for trajectoryidx, trajectory in trajectoriesWithRdpMask.items():
    totalNodes += trajectory.pointCount()
    totalUnmaskedNodes += trajectory.unmaskedCount()
    
print("total: ", str(totalNodes), "\n", "unmasked: ", str(totalUnmaskedNodes), "\n", "ratio: ", str(totalUnmaskedNodes / totalNodes))