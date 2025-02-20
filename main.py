
#### main
def main(config):
    ## Load Dataset


    ## Setup reinforcement learning algorithms (t2vec, etc.)


    ## Setup data collection environment, that is evaluation after each epoch


    ## Main Loop


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