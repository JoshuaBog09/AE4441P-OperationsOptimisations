import gurobipy as gp
import matplotlib.pyplot as plt
import math
import numpy as np


class SquareObsticale:
    def __init__(self, x_lower_left, y_lower_left, x_upper_right, y_upper_right):
        self.x_lower_left = x_lower_left
        self.y_lower_left = y_lower_left
        self.x_upper_right = x_upper_right
        self.y_upper_right = y_upper_right


list_of_obstacles = []

# Limits of the map, width is 600px, height is 400px

model = gp.Model("Path Planning")

# list_of_obstacles.append(SquareObsticale(250, -20, 300, 300))
list_of_obstacles.append(SquareObsticale(330, 30, 400, 410))
# list_of_obstacles.append(SquareObsticale(100, 0, 125, 100))
list_of_obstacles.append(SquareObsticale(250, -10, 300, 150))
# Creation of variables

# initial conditions
v_init = [0, 0]

# max velocities
v_max = 10

# max input
u_max = 0.1

# big number

R = 1e6

# this is equivalent to the number of time steps, L
number_of_time_steps = 220
# how many normals to use

normals = 60

# initial position of the thing
x_begin = (10, 50)

# final position of the robot
x_goal = (250, 300)

# map limits

x_lim = [0, 600]
y_lim = [0, 400]

x = {}
y = {}
ux = {}
uy = {}
b_goal = {}
b_in = {}

for i in range(number_of_time_steps):
    # Continuous variables for the position of the robot
    x[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"x_{i},0", lb=x_lim[0], ub=x_lim[1])
    y[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"x_{i},1", lb=y_lim[0], ub=y_lim[1])
    # Control input vector (math magic)
    ux[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"u_{i},0", lb=-u_max, ub=u_max)
    uy[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"u_{i}, 1", lb=-u_max, ub=u_max)

    # Binary variable to indicate if the goal is reached
    b_goal[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"b_goal_{i}")

for i in range(len(list_of_obstacles)):
    for j in range(4):
        for k in range(number_of_time_steps):
            b_in[i, j, k] = model.addVar(vtype=gp.GRB.BINARY, name=f"b_in_{i},{j},{k}")

model.update()

# Creation of constraints

# Constraint for starting position
model.addLConstr(x[0], '=', x_begin[0], name="C0_x")
model.addLConstr(y[0], '=', x_begin[1], name="C0_y")

# Constraint for starting velocity
model.addLConstr(x[1], '=', x[0] + v_init[0] + ux[0])
model.addLConstr(y[1], '=', y[0] + v_init[1] + uy[0])

# Constrains for ensuring goal is reached
C1 = {}
C2 = {}

for i in range(number_of_time_steps):
    C1[i, 0] = model.addLConstr(x[i] - x_goal[0], '<=', R * (1 - b_goal[i]), name=f"C1_{i},0")
    C1[i, 1] = model.addLConstr(x[i] - x_goal[0], '>=', -R * (1 - b_goal[i]), name=f"C1_{i},1")
    C2[i, 0] = model.addLConstr(y[i] - x_goal[1], '<=', R * (1 - b_goal[i]), name=f"C2_{i},0")
    C2[i, 1] = model.addLConstr(y[i] - x_goal[1], '>=', -R * (1 - b_goal[i]), name=f"C2_{i},1")

model.addLConstr(gp.quicksum(b_goal[i] for i in range(number_of_time_steps)), '=', 1, name="C3")

# # Constraints for ensuring the robot stays in the map
#
# C4 = {}
# C5 = {}
#
# for i in range(number_of_time_steps):
#     C4[i, 0] = model.addLConstr(x[i], '<=', x_lim[1], name=f"C4_{i},0")
#     C4[i, 1] = model.addLConstr(x[i], '>=', x_lim[0], name=f"C4_{i},1")
#     C5[i, 0] = model.addLConstr(y[i], '<=', y_lim[1], name=f"C5_{i},0")
#     C5[i, 1] = model.addLConstr(y[i], '>=', y_lim[0], name=f"C5_{i},1")

# Constraints for ensuring the robot does not collide with the obstacles

C6 = {}
C7 = {}

for i, obstacle in enumerate(list_of_obstacles):
    for j in range(number_of_time_steps):
        C6[i, j, 0] = model.addLConstr(x[j], '<=', obstacle.x_lower_left + R * b_in[i, 0, j], name=f"C6_{i},{j},0")
        C6[i, j, 1] = model.addLConstr(x[j], '>=', obstacle.x_upper_right - R * b_in[i, 1, j], name=f"C6_{i},{j},1")
        C7[i, j, 0] = model.addLConstr(y[j], '<=', obstacle.y_lower_left + R * b_in[i, 2, j], name=f"C7_{i},{j},0")
        C6[i, j, 1] = model.addLConstr(y[j], '>=', obstacle.y_upper_right - R * b_in[i, 3, j], name=f"C7_{i},{j},1")
        model.addLConstr(gp.quicksum(b_in[i, k, j] for k in range(4)), '<=', 3, name="C8")

# Constraints for ensuring the aircraft doesn't break laws of physics
C8 = {}
C9 = {}

# for i in range(number_of_time_steps - 1):
#     C8[i, 0] = model.addLConstr(x[i + 1] - x[i], '<=', v_max_x, name=f"C8_{i},0")
#     C8[i, 1] = model.addLConstr(x[i + 1] - x[i], '>=', -v_max_x, name=f"C8_{i},1")
#     C9[i, 0] = model.addLConstr(y[i + 1] - y[i], '<=', v_max_y, name=f"C9_{i},0")
#     C9[i, 1] = model.addLConstr(y[i + 1] - y[i], '>=', -v_max_y, name=f"C9_{i},1")

# Other constraints for ensuring the aircraft doesn't break laws of physics
for i in range(number_of_time_steps - 1):
    for j in range(normals):
        # Velocity constrain
        C8[i, j] = model.addLConstr(
            (x[i + 1] - x[i]) * math.cos(2 * math.pi / normals * j) + (y[i + 1] - y[i]) * math.sin(
                2 * math.pi / normals * j), '<=', v_max, name=f"C8_{i},{j}")

        # Input constrain
        C9[i, j] = model.addLConstr(
            (ux[i + 1] - ux[i]) * math.cos(2 * math.pi / normals * j) + (uy[i + 1] - uy[i]) * math.sin(
                2 * math.pi / normals * j), '<=', u_max, name=f"C8_{i},{j}")

C10 = {}
C11 = {}

# Correlate input and the velocity
for i in range(number_of_time_steps - 2):
    C10[i] = model.addLConstr(x[i + 2], '=', 2 * x[i + 1] - x[i] + ux[i + 1], name=f"C10_{i}")
    C11[i] = model.addLConstr(y[i + 2], '=', 2 * y[i + 1] - y[i] + uy[i + 1], name=f"C11_{i}")

model.setObjective(gp.quicksum(b_goal[i] * i for i in range(number_of_time_steps)), gp.GRB.MINIMIZE)

model.update()

model.write("models/model.lp")

model.optimize()

x_result = []
y_result = []

for i in range(number_of_time_steps):
    x_result.append(x[i].x)
    y_result.append(y[i].x)


# plt.plot(x_result, y_result, marker='.')

x_plot = []
y_plot = []

for i in range(len(y_result)):
    if math.isclose(y_result[i], x_goal[1], abs_tol=1e-9) and math.isclose(x_result[i], x_goal[0], abs_tol=1e-9):
        break

    x_plot.append(x_result[i])
    y_plot.append(y_result[i])

for obstacle in list_of_obstacles:
    x = [obstacle.x_upper_right, obstacle.x_upper_right,
         obstacle.x_lower_left, obstacle.x_lower_left, obstacle.x_upper_right]
    y = [obstacle.y_upper_right, obstacle.y_lower_left,
         obstacle.y_lower_left, obstacle.y_upper_right, obstacle.y_upper_right]
    plt.plot(x, y)
plt.xlim(x_lim)
plt.ylim(y_lim)

plt.plot(x_plot, y_plot, marker='.', label="Path")

plt.show()
