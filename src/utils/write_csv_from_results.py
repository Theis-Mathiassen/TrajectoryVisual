import pickle
import pandas as pd
import sys

with open('scores.pkl', 'rb') as f:
    data = pickle.load(f)

if not isinstance(data, pd.DataFrame):
    df = pd.DataFrame(data)
else:
    df = data

if __name__ == "__main__":
    filename  = sys.argv[1]
    if filename:
        df.to_csv(filename + '.csv', index=False)
        print("Exported to ", filename)
    else:
        df.to_csv('scores.csv', index=False)
        print("Exported to scores.csv")