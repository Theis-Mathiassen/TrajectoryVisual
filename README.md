# TrajectoryVisual
Our P8 project about visualizing simplified trajectories.

# Setup
Run directly or use docker (Or other containerization tool)
1. Install uv: `pip install uv`
2. Run `uv sync` to set up the environment
3. Download the dataset:
   - Go to: https://www.kaggle.com/datasets/crailtap/taxi-trajectory
   - Download train.csv (Login required)
   - Extract the files to the datasets folder within the repository
   - Copy the first 101 lines into a csv file called small_train.csv
4. Run `uv run main.py` to execute

## Using Docker
Make sure docker engine is installed: https://docs.docker.com/engine/install/
Build the image:
```
docker build -t trajectory .
```
Create and run a container:
```
docker run -dit -e JOB_OUTPUT_DIR=/results/{This runs output directory} --rm --mount type=bind,src=./results/,dst=/results/ trajectory {Command line arguments}
```
Example:
```
docker run -dit -e JOB_OUTPUT_DIR=/results/0 --rm --mount type=bind,src=./results/,dst=/results/ trajectory --knn 1 --range 1 --similarity c
```


