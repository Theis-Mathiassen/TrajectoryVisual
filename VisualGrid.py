import contextily as cx
from rtree import index
import geopandas
import rasterio
from rasterio.plot import show as rioshow
import matplotlib.pyplot as plt
from geodatasets import get_path
from src.load import get_Tdrive
from src.Util import metricToLonLat
from tqdm import tqdm
import random
import sys
sys.path.append("src/")

def getDataDistribution(trajectories, spatialWindow, temporalWindow):
    dataTrajGrid = {}
    dataNodeGrid = {}
    
    for trajectory in tqdm(trajectories.values()):
        for node in trajectory.nodes:
            tup = (node.x // spatialWindow, node.y // spatialWindow, node.t // temporalWindow)
            if tup not in dataTrajGrid:
                dataTrajGrid[tup] = set()
            if tup not in dataNodeGrid:
                dataNodeGrid[tup] = set()
            dataNodeGrid[tup].add((trajectory.id, node.id))
            dataTrajGrid[tup].add(trajectory.id)
    
    return dataTrajGrid, dataNodeGrid


DATABASENAME = 'original_Taxi'

origRtree, origTrajectories = get_Tdrive(filename=DATABASENAME)

trajGrid, NodeGrid = getDataDistribution(origTrajectories, 2000, 10800)

cell = random.choice(list(trajGrid.keys()))

(twest,tsouth,_) = cell
twest -=1
tsouth -=1
teast, tnorth = twest+3, tsouth+3
west, south = metricToLonLat(twest * 2000, tsouth*2000)
east, north = metricToLonLat(teast*2000, tnorth*2000)
"""west, south, east, north = (
    3.616218566894531,
    y * 2000,
    3.8483047485351562,
    51.13994019806845
             )"""
TDrive_img, TDrive_ext = cx.bounds2img(west,
                                     south,
                                     east,
                                     north,
                                     ll=True,
                                     source=cx.providers.CartoDB.Voyager
                                    )

f, ax = plt.subplots(1, figsize=(9, 9))
ax.imshow(TDrive_img, extent=TDrive_ext)
cx.add_basemap(ax)

TrajNodeIds = NodeGrid[cell]

xPoints = []
yPoints = []

for (trajId, nodeId) in TrajNodeIds:
    node = origTrajectories[trajId].nodes[nodeId]
    x,y = metricToLonLat(node.x, node.y)
    
    xPoints.append(x)
    yPoints.append(y)

#plt.plot(xPoints, yPoints, 'o')
plt.show()