import pickle
import sys
from tqdm import tqdm
sys.path.append("src/")
#uv run python print_pkl.py relativepathtopklfile

def print_pickle_file(filepath: str) -> None:
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        lst = 0
        lstNoEndNodes = 0
        empty = 0
        cnt = 0
        total = set()
        repeated = set()
        
        for (q, r) in tqdm(data):
            if len(r) == 0: empty += 1
            else: 
                setTraj = set([trajId for (trajId, _) in r])
                lst += len(setTraj)
                setTrajWithEndNodes = set([trajId for (trajId, nodeId) in r if nodeId == 0])# or nodeId == q.trajectories[q.origin.id].nodes[-1].id
                lstNoEndNodes += len(setTraj.difference(setTrajWithEndNodes))
                repeated.update(total.intersection(r))
                total.update(r)
            cnt += 1
        print(f"total number of nodes {lst}")
        print(f"total number of nodes that aren't initial nodes {lstNoEndNodes}")
        print(f"empty results {empty}")
        print(f"number of results{cnt}")
        avg = lst / cnt
        print(f"average number of trajectories in results {avg}")
        print(f"total number of distinct nodes {len(total)}")
        print(f"total number of nodes that are in more than one result set {len(repeated)}")
        
        Lst = 0
        for (q, r) in tqdm(data):
            setTraj = set([trajId for (trajId, nodeId) in r if (trajId, nodeId) not in repeated])
            Lst += len(setTraj)
            
        print(f"Total result length of trajectories that have distinct nodes {Lst}")
        print(f"average length of this {Lst / cnt}")
        print(f"Average result length of trajectories that have distinct nodes and the repeated nodes {(Lst + len(repeated))/ cnt}")
        
    except Exception as e:
        print(f"Error reading pickle file: {e}")

def minNodesResults(filepath: str) -> None:
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        total = set()
        repeated = set()
        
        for query, res in data:
            total = total.union(res)
            repeated = repeated.union(total.intersection(res))
        print(len(total))
        print(len(repeated))
    except Exception as e:
        print(f"Error reading pickle file: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("need more args. specify dir")
        sys.exit(1)

    pickle_path = sys.argv[1]
    print_pickle_file(pickle_path)