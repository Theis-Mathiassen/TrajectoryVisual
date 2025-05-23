import pickle
import sys
#uv run python print_pkl.py relativepathtopklfile

def print_pickle_file(filepath: str) -> None:
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(data)
    except Exception as e:
        print(f"Error reading pickle file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("need more args. specify dir")
        sys.exit(1)

    pickle_path = sys.argv[1]
    print_pickle_file(pickle_path)