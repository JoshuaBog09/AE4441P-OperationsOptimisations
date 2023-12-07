import gurobipy as gp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import numpy as np


list_of_obstacles = []

# Limits of the map, width is 600px, height is 400px

# Create MILP model.
model = gp.Model("Path Planning")

# Model constants
R = 1e6         # Big number.
NORMALS = 60    # Number of normals used for vector magnitude calcualtion.
DIMENSION = 2   # Number of dimensions.

# Define obstacles.
list_of_obstacles.append(np.array([[150, 200], [200, 410]]))    # Obstacle bounds (dx2 array with [lower_left, upper_right])
list_of_obstacles.append(np.array([[10, -10], [30, 250]]))

# Initial conditions of the vehicle.
x_init = [0, 0]   # Initial position of the vehicle.
v_init = [0, 0]     # Initial velocity of the vehicle.

# Final conditions of the vehicle at the goal.
x_goal = (300, 250) # Goal position

# Equivalent number of time steps, L.
number_of_time_steps = 220

# Limit on parameters
map_bound = np.array([[0, 0], [400, 600]])  # Map bounds (dx2 array with [lower_left, upper_right])
v_max = 10                                  # Maximum velocity (scalar)
u_max = 0.1                                 # Maximum input (scalar)

# Define variable dictionaries.
x = {}          # Decision variables for the position of the rover.
u = {}          # Decision variables for the inputs of the rover.
b_goal = {}     # Boolean indicating whether a node reaches the goal.
b_in = {}       # Boolean used for checking collision with obstacles.

# ================ DEFINE MODEL VARIABLES =========================

# Define the variables for each time step.
for i in range(number_of_time_steps):

    for d in range(DIMENSION):
        # Position variables for the vehicle position, limited by the map bounds.
        x[i, d] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"X[time={i},dim={d}]", lb=map_bound[0, d], ub=map_bound[1, d])

        # Input variables, limited by maximum input.
        u[i, d] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"U[time={i},dim={d}]", lb=-u_max, ub=u_max)

        # Binary for obstacle detection.
        for k in range(2):
            # Binaries for detecting obstacle collision.
            b_in[i, d, k, 0] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_in[time={i},dim={d},obs={k},bound={0}]")
            b_in[i, d, k, 1] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_in[time={i},dim={d},obs={k},bound={1}]")

    # Binary indicating first node that reaches goal.
    b_goal[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_goal_{i}")

model.update()

# =========== CREATE MODEL CONSTRAINTS ==========================

# Constraints for forcing the initial conditions.
CONST_X_INIT = {}
CONST_V_INIT = {}

for d in range(DIMENSION):
    # Initial position constraint
    CONST_X_INIT[d] = model.addLConstr(x[0, d], '=', x_init[d], name=f"CONST_X_INIT[dim={d}]")

    # Initial velocity constraint
    CONST_V_INIT[d] = model.addLConstr(x[1, d], '=', x[0, d] + v_init[d] + u[0, d], name=f"CONST_V_INIT[dim={d}]")

# Constraints for detecting when a node reaches the goal.
CONST_X_GOAL = {}

for i in range(number_of_time_steps):
    for d in range(DIMENSION):
        CONST_X_GOAL[i, d, 0] = model.addLConstr(x[i, d] - x_goal[d], '<=', R * (1 - b_goal[i]), name=f"CONST_X_GOAL[time={i},dim={d},const=0]")
        CONST_X_GOAL[i, d, 1] = model.addLConstr(x[i, d] - x_goal[d], '>=', -R * (1 - b_goal[i]), name=f"CONST_X_GOAL[time={i},dim={d},const=1]")

# Constraint for ensuring exactly one node reaches the goal.
CONST_REACH_GOAL = model.addLConstr(gp.quicksum(b_goal[i] for i in range(number_of_time_steps)), '=', 1, name="CONST_REACH_GOAL")

# Constraints for ensuring the robot does not collide with the obstacles.
CONST_OBSTACLE = {}
CONST_OBSTACLE_DETECTION = {}

for i in range(number_of_time_steps):

    for k, obstacle in enumerate(list_of_obstacles):

        for d in range(DIMENSION):

            CONST_OBSTACLE[i, d, k, 0] = model.addLConstr(x[i, d], '<=', obstacle[0, d] + R * b_in[i, d, k, 0], name=f"CONST_OBSTACLE[time={i},dim={d},obs={k},bound={0}]")
            CONST_OBSTACLE[i, d, k, 1] = model.addLConstr(x[i, d], '>=', obstacle[1, d] - R * b_in[i, d, k, 1], name=f"CONST_OBSTACLE[time={i},dim={d},obs={k},bound={1}]")

        CONST_OBSTACLE_DETECTION[i, k] = model.addLConstr(gp.quicksum(b_in[i, d, k, 0] + b_in[i, d, k, 1] for d in range(DIMENSION)), '<=', 3, name=f"CONST_OBSTACLE_DETECTION[time={i},obs={k}]")

# Constraints for ensuring the aircraft doesn't break laws of physics
CONST_V_MAX = {}
CONST_U_MAX = {}

# Other constraints for ensuring the aircraft doesn't break laws of physics
for i in range(number_of_time_steps - 1):

    for n in range(NORMALS):
        # Velocity constrain
        CONST_V_MAX[i, n] = model.addLConstr(
            (x[i + 1, 0] - x[i, 0]) * np.cos(2 * np.pi / NORMALS * n) + (x[i + 1, 1] - x[i, 1]) * np.sin(
                2 * np.pi / NORMALS * n), '<=', v_max, name=f"COST_V_MAX[time={i},normal={n}]")

        # Input constrain
        CONST_U_MAX[i, n] = model.addLConstr(
            (u[i + 1, 0] - u[i, 0]) * np.cos(2 * np.pi / NORMALS * n) + (u[i + 1, 1] - u[i, 1]) * np.sin(
                2 * np.pi / NORMALS * n), '<=', u_max, name=f"ONST_U_MAX[time={i},normals={n}]")

# Relation input, velocity and position next point.
CONST_POS = {}

# Correlate input and the velocity
for i in range(number_of_time_steps - 2):
    for d in range(DIMENSION):
        CONST_POS[i, d] = model.addLConstr(x[i + 2, d], '=', 2 * x[i + 1, d] - x[i, d] + u[i + 1, d], name=f"CONST_POS[time={i},dim={d}]")

OBJECTIVE = model.setObjective(gp.quicksum(b_goal[i] * i for i in range(number_of_time_steps)), gp.GRB.MINIMIZE)

model.update()

model.write("models/model.lp")

model.optimize()

# ============== PLOT RESULTS =================================

# Define figure and axis.
fig, ax = plt.subplots()

# Set limits
ax.set_xlim(map_bound[0, 0], map_bound[1, 0])
ax.set_ylim(map_bound[0, 1], map_bound[1, 1])

# Construct list with vehicle location to be plotted
x_plot = []
y_plot = []

for i in range(number_of_time_steps):
    x_plot.append(x[i, 0].X)
    y_plot.append(x[i, 1].X)

    # Stop plot once goal is reached.
    if b_goal[i].X == 1:
        break

# Plot vehicle location
ax.plot(x_plot, y_plot, marker=".", color='red', label="Path")
ax.quiver(x_plot, y_plot, marker=".", color='k', label="Path")


# Plot obstacles
for obstacle in list_of_obstacles:
    origin = obstacle[0]
    delta = obstacle[1] - obstacle[0]
    width = delta[0]
    height = delta[1]

    ax.add_patch(Rectangle(origin, width, height, color='dimgrey'))

#display plot
plt.show()
