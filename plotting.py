import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from receding_horizon import Scene, Results, Config
import numpy as np
import pickle


def static_plot(scene: Scene, results: Results):
    # Plot the scene
    fig, ax = plt.subplots()

    # Set limits
    ax.set_xlim(scene.map_bounds[0, 0], scene.map_bounds[1, 0])
    ax.set_ylim(scene.map_bounds[0, 1], scene.map_bounds[1, 1])

    # Plot vehicle location
    # ax.plot(x_path, y_path, marker=".", color='red', label="Path")
    ax.plot(scene.goal[0], scene.goal[1], marker="X", color='green', label="Goal")

    # Plot obstacles
    for obstacle in scene.obstacles:
        origin = obstacle[0]
        delta = obstacle[1] - obstacle[0]
        width = delta[0]
        height = delta[1]

        ax.add_patch(Rectangle(origin, width, height, color='dimgrey'))

    for path in results.plan_path:
        ax.plot(path[:, 0], path[:, 1], marker="", color='blue', linewidth=0.33, zorder=0)

    results.exec_path = np.concatenate(results.exec_path)

    ax.plot(results.exec_path[:, 0], results.exec_path[:, 1], marker=".", color='red', linewidth=1, zorder=10,
            label="Executed Path", markersize=5.5, markevery=3)

    # display plot
    plt.legend()
    plt.show()


def animate(i, ax, results: Results, config: Config):
    ax.plot(results.plan_path[i][:, 0], results.plan_path[i][:, 1], marker="", color='blue', linewidth=0.33, zorder=0)

    if i == 0:
        ax.plot(results.exec_path[i][:, 0], results.exec_path[i][:, 1], marker=".", color='red', linewidth=1, zorder=10)
    else:
        path = np.vstack((results.exec_path[i-1][-1, :], results.exec_path[i]))
        ax.plot(path[:, 0], path[:, 1], marker=".", color='red', linewidth=1, zorder=10)


def animated_plot(scene: Scene, results: Results, config: Config):
    fig, ax = plt.subplots()
    # Set limits
    ax.set_xlim(scene.map_bounds[0, 0], scene.map_bounds[1, 0])
    ax.set_ylim(scene.map_bounds[0, 1], scene.map_bounds[1, 1])

    # Plot vehicle location
    # ax.plot(x_path, y_path, marker=".", color='red', label="Path")
    ax.plot(scene.goal[0], scene.goal[1], marker="X", color='green', label="Goal")

    # Plot obstacles
    for obstacle in scene.obstacles:
        origin = obstacle[0]
        delta = obstacle[1] - obstacle[0]
        width = delta[0]
        height = delta[1]

        ax.add_patch(Rectangle(origin, width, height, color='dimgrey'))
    ani = FuncAnimation(fig, animate, fargs=(ax, results, config), frames=len(results.exec_path))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # static_plot(scene, results)
    dir = ('pickle_data/2024_02_29-04_24_29_PM')

    # Load the data
    try:
        with open(f'{dir}/scene.pickle', 'rb') as f:
            scene = pickle.load(f)
    except FileNotFoundError:
        print("Scene pickle file not found. Please run the simulation first.")

    try:
        with open(f'{dir}/results.pickle', 'rb') as f:
            results = pickle.load(f)
    except FileNotFoundError:
        print("Results pickle file not found. Please run the simulation first.")

    try:
        with open(f'{dir}/config.pickle', 'rb') as f:
            config = pickle.load(f)
    except FileNotFoundError:
        print("Config pickle file not found. Please run the simulation first.")
    animated_plot(scene, results, config)
