from src.evaluation import getAverageF1ScoreAll, GetSimplificationError
from load import build_Rtree

#### main
def main(config):
    ## Load Dataset

    build_Rtree("trimmed_small_train.csv", "original_Tdrive")
    build_Rtree("trimmed_small_train.csv", "simplified_Tdrive")
    ## Setup reinforcement learning algorithms (t2vec, etc.)


    ## Setup data collection environment, that is evaluation after each epoch


    ## Main Loop

        # While above compression rate
        
        # Generate and apply queries, giving scorings to points

        # Remove x points with the fewest points

        # Collect evaluation data
            # getAverageF1ScoreAll, GetSimplificationError

    ## Save results


    ## Plot models

    pass




if __name__ == "__main__":
    config = {}
    config["epochs"] = 100                  # Number of epochs to simplify the trajectory database
    config["compression_rate"] = 0.1        # Compression rate of the trajectory database
    config["DB_size"] = 100                 # Amount of trajectories to load (Potentially irrelevant)
    config["verbose"] = True                # Print progress

    main(config)