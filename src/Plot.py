import numpy as np
import matplotlib.pyplot as plt
import Trajectory as T

def plotTrajectory(traj: T):
    # Plots a single trajectory.
    # Input is assumed to be a 'Trajectory' instance.
    xs = []
    ys = []

    for elem in traj.nodes:
        xs.append(elem.x)
        ys.append(elem.y)

    plt.plot(xs, ys)
    plt.show()

def plotTrajectories(traj: list[T]):
    # Plots multiple trajectories.
    # Input is assumed to be a list of 'Trajectory' instances.
    for Trajectory in traj:
        xs = []
        ys = []

        for elem in Trajectory.nodes:
            xs.append(elem.x)
            ys.append(elem.y)

        plt.plot(xs, ys)
    plt.show()