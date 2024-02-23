import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from receding_horizon import Scene, Results
import numpy as np
import pickle

dir = "pickle_data/2024_02_23-01_48_33_PM"

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

ax.plot(results.exec_path[:, 0], results.exec_path[:, 1], marker=".", color='red', linewidth=1, zorder=10, label="Executed Path", markersize=5.5, markevery=3)

# display plot
plt.legend()
plt.show()