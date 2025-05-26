import pickle
import sys
sys.path.append("src/")
#uv run python print_pkl.py relativepathtopklfile

def print_pickle_file(filepath: str) -> None:
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        empty = 0
        cnt = 0
        for (q, r) in data:
            empty += len(set([trajId for (trajId, _) in r]))
            cnt += 1
        print(empty)
        avg = empty / cnt
        print(avg)
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