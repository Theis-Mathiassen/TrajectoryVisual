from rtree import index
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import re

import shutil
import copy

from ast import literal_eval
import json
from src.Util import lonLatToMetric
from tqdm import tqdm

from src.Node import Node
from src.Trajectory import Trajectory
from src.Filter import Filter

CHUNKSIZE = 10**5
PAGESIZE = 16000


# Handle frequent problem that occurs with corrupted load
def checkRtreeIndexEmpty(filename):
    if os.path.exists(filename + '.index'):
        if os.path.getsize(filename + '.index') == 0:
            print("Found issue with old rtree deleting before load...")
            os.remove(filename + ".index")
            os.remove(filename + ".data")

def checkCurrentRtreeMatches(Rtree, trajectories, filename):
    """
    Function to check if rtree matches trajectories. This happens if several csv files are loaded under the same name.
    We do this by checking for 10 random nodes from the trajectories
    """

    # If Rtree does not match Trajectories, delete Rtree and create a new one
    if not checkCurrentRtreeMatchesHelper(Rtree, trajectories):
        print("Rtree does not match Trajectories, deleting Rtree and creating a new one...")
        Rtree.close()
        #del Rtree # This deletes the old Rtree from program memory, allowing us to delete the index and data files
        os.remove(filename + '.index')
        os.remove(filename + '.data')
        Rtree, _ = loadRtree(filename, trajectories)

    return Rtree

def checkCurrentRtreeMatchesHelper(Rtree, trajectories) -> bool:
    trajectoriesList = list(trajectories.values())
    for trajectory in trajectoriesList[:10]:
        firstNode = trajectory.nodes[0]

        hits = list(Rtree.intersection((firstNode.x, firstNode.y, firstNode.t, firstNode.x, firstNode.y, firstNode.t), objects="raw"))

        if len(hits) == 0:
            return False

        flag = False
        actualTrajectoryId = trajectory.id
        actualNodeId = firstNode.id
        for hit in hits:
            trajectory_id, node_id = hit

            # If the trajectory id and node id match then we have a match
            if trajectory_id == actualTrajectoryId and node_id == actualNodeId:
                flag = True
                break
        # If no matches then return false
        if not flag:
            return False
        
    # If all matches then return true
    return True

def get_Tdrive(filename="") :
    cwd = os.getcwd()
    path = os.path.join(cwd, 'datasets', 'Tdrive.csv')
    if os.path.exists(path):
        print("Tdrive already loaded to CSV, skipping load from folder...")
    else:
        tDriveToCsv()

    checkRtreeIndexEmpty(filename=filename)

    Rtree, Trajectories = load_Tdrive_Rtree(filename=filename)



    # If Rtree does not match Trajectories, delete Rtree and create a new one
    Rtree = checkCurrentRtreeMatches(Rtree, Trajectories, filename)


    return Rtree, Trajectories


def load_Tdrive(src : str, filename="") : 

    cwd = os.getcwd()

    #Refactor to use a chunksize instead!
    df = pd.read_csv(os.path.join(cwd, 'datasets', src), delimiter=',')

    #Preprocessing 
    df = df.drop(columns=['CALL_TYPE','ORIGIN_CALL','ORIGIN_STAND','TAXI_ID', 'DAY_TYPE'])
    
   
    tqdm.pandas(desc="Evaluating polyline")
    df["POLYLINE"] = df["POLYLINE"].progress_apply(json.loads)
    

    droppedRows = []
    for index in range(len(df)) : 
        if df['MISSING_DATA'][index] == "True" or len(df['POLYLINE'][index]) == 0:
            droppedRows.append(index)

    df = df.drop(columns=['MISSING_DATA'])
    df = df.drop(droppedRows)
        
    tqdm.pandas(desc="Converting lon lat to metric")
    df["POLYLINE"] = df["POLYLINE"].progress_apply(rowLonLatToMetric)
    
    #Save trimmed data 
    if filename == '' : 
        if os.path.exists(os.path.join(cwd, 'datasets', 'default.csv')) : 
            os.remove(os.path.join(cwd, 'datasets', 'default.csv'))
        df.to_csv(path_or_buf=os.path.join(cwd, 'datasets', 'default.csv'), index=False)
    else :
        if os.path.exists(os.path.join(cwd, 'datasets', filename)) :
            os.remove(os.path.join(cwd, 'datasets', filename))
        df.to_csv(path_or_buf=os.path.join(cwd, 'datasets', filename), index=False)
    
def jsonLoadsNumpy(polylineString) :
    if pd.isna(polylineString) or not isinstance(polylineString, str) :
        return []
    

    # This regex code is copied from online sources

    # Remove `np.float64(...)` but keep values
    cleanString = re.sub(r'np\.float64\(([^)]+)\)', r'\1', polylineString)

    # Convert Python-style tuples `(x, y, z)` to JSON-style lists `[x, y, z]`
    cleanString = cleanString.replace("(", "[").replace(")", "]")

    # Fix any trailing commas inside the list
    cleanString = re.sub(r',\s*([\]\}])', r'\1', cleanString)

    # Ensure valid JSON format by replacing single quotes with double quotes (if present)
    cleanString = cleanString.replace("'", '"')

    # If the string isn't enclosed in brackets, add them
    if not cleanString.startswith("["):
        cleanString = "[" + cleanString
    if not cleanString.endswith("]"):
        cleanString = cleanString + "]"

    # debugging....
    try:
        parsed_json = json.loads(cleanString)
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print(f"Problematic string: {cleanString}")
        return []  # Return empty list instead of crashing




def load_Tdrive_Rtree(filename=""):
    dataset = "TDrive.csv"
    # Read csv file as dataframe
    tqdm.pandas()
    cwd = os.getcwd()
    path = os.path.join(cwd, 'datasets', dataset)


    df = pd.read_csv(path, converters={'POLYLINE' : jsonLoadsNumpy, 'TRIP_ID' : json.loads})
    
    
    # Set up properties
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    p.leaf_capacity = 1000
    p.pagesize = PAGESIZE
    #p.filename = filename

    """ if filename=='' :
        print("No filename!")
        Rtree_ = index.Index(properties=p)
    else :
        if os.path.exists(filename+'.dat'):
            os.remove(filename+'.dat')
            print('remove', filename+'.dat')
        if os.path.exists(filename+'.idx'):
            os.remove(filename+'.idx')
            print('remove', filename+'.idx') 
    
    print("Eval polyline...")
    df["POLYLINE"] = df["POLYLINE"].progress_apply(json.loads)
    print("Done!") """
    
    polylines = np.array(df['POLYLINE'])
    trip_ids = np.array(df['TRIP_ID'])
    
    if os.path.exists(filename + '.index'):
        Rtree_ = index.Index(filename, properties=p)
    else:
        Rtree_ = index.Index(filename, TDriveDataStream(polylines, trip_ids), properties=p)
    
    print("Creating trajectories..")
    c = 0
    delete_rec = {}
    Trajectories = {}
    length = len(trip_ids)
    for i in tqdm(range(length)):
        c = 0
        nodes = []
        for x, y, t in polylines[i]:
            nodes.append(Node(c, x, y, t))
            c += 1
        Trajectories.update({int(trip_ids[i]) : Trajectory(int(trip_ids[i]), ma.copy(nodes))})        
    
        
    return Rtree_, Trajectories



#Build a rtree from csv file and the corresponding trajectories. It is assumed that the csv file contains columns 'TIMESTAMP', 'TRIP_ID' and 'POLYLINE'.
#Function is in two parts: Creating the rtree and creating trajectories.
#Creating the rtree: Happens via bulk load through generator function (datastream). This is a lot faster than inserting each point.
#Creating trajectories: Create a unique trajectory with id = 'TRIP_ID' and nodes = 'POLYLINE' -> Nodes objects
#TO DO:
#Include pagesize param (property of p) for optimizing rtree accesses.
#Reading the csv in chunks if possible would maybe improve performance, unless this hinders bulk loading the rtree.
def build_Rtree(dataset, filename='') :
    # Read csv file as dataframe
    tqdm.pandas()
    cwd = os.getcwd()
    path = os.path.join(cwd, 'datasets', dataset)
    df = pd.read_csv(path, converters={'POLYLINE' : json.loads, 'TRIP_ID' : json.loads, 'TIMESTAMP' : json.loads})
    
    # Set up properties
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    p.leaf_capacity = 1000
    p.pagesize = PAGESIZE
    #p.filename = filename
    
    polylines = np.array(df['POLYLINE'])
    timestamps = np.array(df['TIMESTAMP'])
    trip_ids = np.array(df['TRIP_ID'])
    
    
    if os.path.exists(filename + '.index'):
        Rtree_ = index.Index(filename, properties=p)
    else:
        Rtree_ = index.Index(filename, datastream(polylines, timestamps, trip_ids), properties=p)
    
    
    #print("Creating trajectories..")
    c = 0
    delete_rec = {}
    Trajectories = {}
    length = len(trip_ids)
    for i in tqdm(range(length), desc="Creating trajectories"):
        if len(polylines[i]) == 0:
            pass
        else:
            t = 0
            c = 0
            nodes = [] #np.array(nparrayStream(polylines[i], timestamps[i]))
            for x, y in polylines[i]:
                nodes.append(Node(c, x, y, timestamps[i] + t*15))
                c += 1
                t += 1
            Trajectories.update({int(trip_ids[i]) : Trajectory(int(trip_ids[i]), ma.copy(nodes))})        
    
        
    return Rtree_, Trajectories



def load_Geolife(src_dir: str, filename: str = "", max_trajectories: int = None, target_area: tuple = None):
    """
    Load and process the Geolife dataset.
    
    Args:
        src_dir: Directory containing the Geolife dataset
        filename: Output filename for processed data
        max_trajectories: Maximum number of trajectories to load
        target_area: Tuple of (min_lat, max_lat, min_lon, max_lon) to filter trajectories by geographic area
    """
    cwd = os.getcwd()
    base_path = os.path.join(cwd, 'src', src_dir)
    
    # Initialize lists to store trajectory data
    all_trajectories = []
    trajectory_count = 0
    
    # process each of the 183 users' directory
    for user_dir in tqdm(os.listdir(base_path), desc="Processing user directories"):
        if not user_dir.isdigit():
            continue
            
        user_path = os.path.join(base_path, user_dir, 'Trajectory')
        if not os.path.exists(user_path):
            continue
            
        # process each .plt file in the user's trajectory directory
        for plt_file in os.listdir(user_path):
            if not plt_file.endswith('.plt'):
                continue
                
            file_path = os.path.join(user_path, plt_file)
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # skip headers 
            if len(lines) < 7:
                continue
                
            # Parse trajectory points
            points = []
            for line in lines[6:]:  # Skip header lines
                try:
                    lat, lon, _, _, _, _, _ = line.strip().split(',')
                    lat, lon = float(lat), float(lon)
                    
                    # Convert to metric coordinates
                    x, y = lonLatToMetric(lon, lat)
                    points.append([x, y])
                except:
                    continue
                    
            if not points:
                continue
                
            # check if trajectory is in target area if specified
            if target_area:
                min_lat, max_lat, min_lon, max_lon = target_area
                if not any(min_lat <= lat <= max_lat and min_lon <= lon <= max_lon 
                          for lat, lon in [(p[0], p[1]) for p in points]):
                    continue
            
            # Add trajectory to list
            all_trajectories.append({
                'TRIP_ID': trajectory_count,
                'TIMESTAMP': int(os.path.splitext(plt_file)[0]),
                'POLYLINE': points
            })
            trajectory_count += 1
            
            # Check if we've reached max_trajectories
            if max_trajectories and trajectory_count >= max_trajectories:
                break
                
        if max_trajectories and trajectory_count >= max_trajectories:
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trajectories)
    
    # Save processed data
    if filename == '':
        if os.path.exists(os.path.join(cwd, 'datasets', 'default.csv')):
            os.remove(os.path.join(cwd, 'datasets', 'default.csv'))
        df.to_csv(path_or_buf=os.path.join(cwd, 'datasets', 'default.csv'), index=False)
    else:
        if os.path.exists(os.path.join(cwd, 'datasets', filename)):
            os.remove(os.path.join(cwd, 'datasets', filename))
        df.to_csv(path_or_buf=os.path.join(cwd, 'datasets', filename), index=False)
    
    return df

#Build a rtree from csv file and the corresponding trajectories. It is assumed that the csv file contains columns 'TIMESTAMP', 'TRIP_ID' and 'POLYLINE'.
#Function is in two parts: Creating the rtree and creating trajectories.
#Creating the rtree: Happens via bulk load through generator function (datastream). This is a lot faster than inserting each point.
#Creating trajectories: Create a unique trajectory with id = 'TRIP_ID' and nodes = 'POLYLINE' -> Nodes objects
#TO DO:
#Include pagesize param (property of p) for optimizing rtree accesses.
#Reading the csv in chunks if possible would maybe improve performance, unless this hinders bulk loading the rtree.
def build_Rtree(dataset, filename='') :
    # Read csv file as dataframe
    tqdm.pandas()
    cwd = os.getcwd()
    path = os.path.join(cwd, 'datasets', dataset)
    df = pd.read_csv(path, converters={'POLYLINE' : json.loads, 'TRIP_ID' : json.loads, 'TIMESTAMP' : json.loads})
    
    # Set up properties
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    p.leaf_capacity = 1000
    p.pagesize = PAGESIZE
    #p.filename = filename
    
    polylines = np.array(df['POLYLINE'])
    timestamps = np.array(df['TIMESTAMP'])
    trip_ids = np.array(df['TRIP_ID'])

    checkRtreeIndexEmpty(filename=filename)
    
    if os.path.exists(filename + '.index'):
        Rtree_ = index.Index(filename, properties=p)
    else:
        Rtree_ = index.Index(filename, datastream(polylines, timestamps, trip_ids), properties=p)
    
    
    #print("Creating trajectories..")
    c = 0
    delete_rec = {}
    Trajectories = {}
    length = len(trip_ids)
    for i in tqdm(range(length), desc="Creating trajectories"):
        if len(polylines[i]) == 0:
            pass
        else:
            t = 0
            c = 0
            nodes = [] #np.array(nparrayStream(polylines[i], timestamps[i]))
            for x, y in polylines[i]:
                nodes.append(Node(c, x, y, timestamps[i] + t*15))
                c += 1
                t += 1
            Trajectories.update({int(trip_ids[i]) : Trajectory(int(trip_ids[i]), ma.copy(nodes))})        
    
    # Check if the current rtree matches the trajectories
    Rtree_ = checkCurrentRtreeMatches(Rtree_, Trajectories, filename)

    return Rtree_, Trajectories

#Convert a polyline row in lon lat coordinates to metric coordinates
def rowLonLatToMetric(row):
    nodes = []
    for x, y in row:
        nodes.append(list(lonLatToMetric(x, y)))
    return nodes

def loadDatasetWithFilters(rtreeName : str, dataset, filters : list[Filter]):
    """ Applies a list of filters in sequential order to the data """
    trajectories = loadDataTrajectories(dataset)

    for filter in filters:
        filter.filterTrajectories(trajectories)
    
    Rtree_, _ = loadRtree(rtreeName, trajectories) # Converts trajectories to rtree

    return Rtree_, trajectories
    
def loadDataTrajectories(dataset):
    """ Loads the trajectories from a dataset """
    # Read csv file as dataframe
    tqdm.pandas()
    cwd = os.getcwd()
    path = os.path.join(cwd, 'datasets', dataset)
    df = pd.read_csv(path, converters={'POLYLINE' : json.loads, 'TRIP_ID' : json.loads, 'TIMESTAMP' : json.loads})
    
    # Set up properties
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    p.leaf_capacity = 1000
    p.pagesize = PAGESIZE
    #p.filename = filename
    
    polylines = np.array(df['POLYLINE'])
    timestamps = np.array(df['TIMESTAMP'])
    trip_ids = np.array(df['TRIP_ID'])

    c = 0
    Trajectories = {}
    length = len(trip_ids)
    for i in tqdm(range(length), desc="Creating trajectories"):
        if len(polylines[i]) == 0:
            pass
        else:
            t = 0
            c = 0
            nodes = [] #np.array(nparrayStream(polylines[i], timestamps[i]))
            for x, y in polylines[i]:
                nodes.append(Node(c, x, y, timestamps[i] + t*15))
                c += 1
                t += 1
            Trajectories.update({int(trip_ids[i]) : Trajectory(int(trip_ids[i]), ma.copy(nodes))})        
    

#Function to load existing rtree and create a copy of it.
#TO DO:
#Fix it. Dont think it works, but maybe it isn't neccesary?
def loadRtree(rtreeName : str, trajectories):
    p = index.Property()
    p.dimension = 3
    p.dat_extension = 'data'
    p.idx_extension = 'index'
    p.leaf_capacity = 1000
    p.pagesize = PAGESIZE
    #p.filename = rtreeName
    #p.overwrite = True
    #bounds = originalRtree.bounds
    #points = list(originalRtree.intersection((bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]), objects=True))
    rtreeCopy = index.Index(rtreeName, pointStream(trajectories), properties=p)
    #rtreeCopy.insert(pointStream(originalRtree))
    trajectoriesCopy = copy.deepcopy(trajectories)
    return rtreeCopy, trajectoriesCopy

    
#Generator function taking a dataframe with columns 'TIMESTAMP', 'TRIP_ID' and 'POLYLINE'.
#Yields a rtree point for each point in each polyline
def datastream(polylines, timestamps, trip_ids):
    length = len(trip_ids)
    c = 0
    for i in tqdm(range(length), total=length, desc="Loading trajectories into rtree") :
        t = 0
        timestamp = timestamps[i]
        if len(polylines[i]) == 0:
            pass
        else:
            for x, y in polylines[i] :
                obj=(int(trip_ids[i]), t)
                curTimestamp = timestamp + (15*t)
                yield (c, (x, y, curTimestamp, x, y, curTimestamp), obj)
                
                c+=1
                t+=1

def TDriveDataStream(polylines, trip_ids) :
    c = 0
    length = len(trip_ids)
    for i in tqdm(range(length), total=length, desc="Loading trajectories into rtree") :
        idx = 0
        if len(polylines[i]) == 0:
            pass
        else:
            for x, y, t in polylines[i] :
                obj=(int(trip_ids[i]), idx)
                yield (c, (x, y, t, x, y, t), obj)
                c+=1
                idx+=1

#Generator function taking a rtree
#Yields all points of the rtree 
def pointStream(trajectories: dict):
    for trajectory in tqdm(trajectories.values(), total=len(trajectories.values()), desc="Loading trajectories into rtree"):
        nodes = trajectory.nodes.compressed()
        for i in range(len(nodes)):
            obj=(trajectory.id, nodes[i].id)
            yield (trajectory.id, (nodes[i].x, nodes[i].y, nodes[i].t, nodes[i].x, nodes[i].y, nodes[i].t), obj)
    

def nparrayStream(polyline, timestamp):
    c = 0
    t = 0
    for x, y in polyline:
            yield Node(c, x, y, timestamp + t*15)
            c += 1
            t += 1
def tDriveToCsv():
    """
    Function to convert the T-Drive dataset into a single csv file with TRIP_ID, POLYLINE columns.
    """
    cwd = os.getcwd()

    directory = os.fsencode(os.path.join(cwd, 'datasets','taxi_log_2008_by_id'))
        
    csvdf = pd.DataFrame(columns=['TRIP_ID', 'POLYLINE'])

    taxiIdx = 0

    super_x = []

    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        
        #df = pd.read_csv(os.path.join(os.fsdecode(directory),filename), header=None, delimiter=',', names=['id', 'time', 'lon', 'lat'], parse_dates=['time'], date_format="%Y%m%d%H%M%S") #, date_format="%Y%m%d%H%M%S"
        df = pd.read_csv(os.path.join(os.fsdecode(directory),filename), header=None, delimiter=',', names=['id', 'time', 'lon', 'lat'])

        df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S", errors='coerce') # Set invalid values to NaN
        
        rows_before = len(df)

        # Drops all rows with missing data(NaN) in the time column
        df.dropna(subset=['time'], inplace=True)

        rows_after = len(df)

        if rows_after < rows_before:
            print(f"Dropped {rows_before-rows_after} rows from {filename}")

        if len(df) == 0: 
            continue
        df['time'] = pd.to_datetime(df['time'], yearfirst=True).astype(int) // 10**9
        df.astype({'lon' : 'Float64', 'lat' : 'Float64'})
        polyline = df.apply(lambda row : (row['lon'], row['lat'], row['time']), axis=1)
        super_x.append([df['id'][0], polyline.tolist()])
        #csvdf = pd.concat([pd.DataFrame({'TRIP_ID' : df['id'][0], 'POLYLINE' : polyline}, columns=['TRIP_ID', 'POLYLINE']), csvdf], ignore_index=True)
        #csvdf.loc[taxiIdx] = [df['id'][0], polyline]
        taxiIdx += 1
        
    csvdf = pd.concat([csvdf, pd.DataFrame(super_x, columns=['TRIP_ID', 'POLYLINE'])], ignore_index=True, axis=0)
    csvdf.to_csv(path_or_buf=os.path.join(cwd, 'datasets', 'TDrive.csv'), sep=',', index=False)

def datastreamTriple(polylines, trip_ids):
    """
    datastream where polylines is an array of (x, y, t) coordinates.

    Args:
        polylines ([(x : float, y : float, t : float)..]): Array of coordinates  
        trip_ids (int): Trip id / trajectory id.

    Yields:
        Index.item(): tuple representing the rtree item with object
    """
    length = len(trip_ids)
    for i in tqdm(range(length)) :
        c = 0
        for x, y, t in polylines[i] :
            obj=(trip_ids[i], c)
            yield (c, (x, y, t, x, y, t), obj)
            c+=1




if __name__ == "__main__":
    tDriveToCsv()
    """ load_Tdrive("trimmed_small_train.csv")
    Rtree_, Trajectories = build_Rtree("trimmed_small_train.csv", "test")

    hits = list(Rtree_.intersection((-8.66,41.13, 1372636858-2, -8.618643,41.17, 1372637303+100), objects=True))
    print("(Trajectory ID, Node id) pair for intersecting trajectories on range query : ")
    print([(n.object) for n in hits]) """

