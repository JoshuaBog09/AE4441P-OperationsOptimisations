import gurobipy as gp
import matplotlib.pyplot as plt


class SquareObsticale:
    def __init__(self, x_lower_left, y_lower_left, x_upper_right, y_upper_right):
        self.x_lower_left = x_lower_left
        self.y_lower_left = y_lower_left
        self.x_upper_right = x_upper_right
        self.y_upper_right = y_upper_right


list_of_obstacles = []

# Limits of the map, width is 600px, height is 400px

model = gp.Model("Path Planning")

list_of_obstacles.append(SquareObsticale(250, 150, 350, 250))

# Creation of variables

# max velocities

v_max_x = 5
v_max_y = 5

# big number

R = 1e6

# this is equivalent to the number of time steps, L
number_of_time_steps = 122

# initial position of the thing
x_begin = (0, 0)

# final position of the robot
x_goal = (600, 200)

x = {}
y = {}
b_goal = {}
b_in = {}

for i in range(number_of_time_steps):
    # Continuous variables for the position of the robot
    x[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"x_{i},0")
    y[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"x_{i},1")

    # Binary variable to indicate if the goal is reached
    b_goal[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"b_goal_{i}")

for i in range(len(list_of_obstacles)):
    for j in range(4):
        b_in[i, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"b_in_{i},{j}")

model.update()

# Creation of constraints

# Constraint for starting position
model.addLConstr(x[0], '=', x_begin[0], name="C0_x")
model.addLConstr(y[0], '=', x_begin[1], name="C0_y")

# Constrains for ensuring goal is reached
C1 = {}
C2 = {}

for i in range(number_of_time_steps):
    C1[i, 0] = model.addLConstr(x[i] - x_goal[0], '<=', R * (1 - b_goal[i]), name=f"C1_{i},0")
    C1[i, 1] = model.addLConstr(x[i] - x_goal[0], '>=', -R * (1 - b_goal[i]), name=f"C1_{i},1")
    C2[i, 0] = model.addLConstr(y[i] - x_goal[1], '<=', R * (1 - b_goal[i]), name=f"C2_{i},0")
    C2[i, 1] = model.addLConstr(y[i] - x_goal[1], '>=', -R * (1 - b_goal[i]), name=f"C2_{i},1")

model.addConstr(gp.quicksum(b_goal[i] for i in range(number_of_time_steps)), '=', 1, name="C3")

# Constraints for ensuring the robot stays in the map

C4 = {}
C5 = {}

for i in range(number_of_time_steps):
    C4[i, 0] = model.addLConstr(x[i], '<=', x_goal[0], name=f"C4_{i},0")
    C4[i, 1] = model.addLConstr(x[i], '>=', 0, name=f"C4_{i},1")
    C5[i, 0] = model.addLConstr(y[i], '<=', x_goal[1], name=f"C5_{i},0")
    C5[i, 1] = model.addLConstr(y[i], '>=', 0, name=f"C5_{i},1")

# Constraints for ensuring the robot does not collide with the obstacles

C6 = {}
C7 = {}

for i, obstacle in enumerate(list_of_obstacles):
    C6[i, 0] = model.addLConstr(x[i], '<=', obstacle.x_lower_left + R * b_in[i, 0], name=f"C6_{i},0")
    C6[i, 1] = model.addLConstr(x[i], '>=', obstacle.x_upper_right - R * b_in[i, 1], name=f"C6_{i},1")
    C7[i, 0] = model.addLConstr(y[i], '<=', obstacle.y_lower_left + R * b_in[i, 2], name=f"C7_{i},0")
    C6[i, 1] = model.addLConstr(y[i], '>=', obstacle.y_upper_right - R * b_in[i, 3], name=f"C7_{i},1")
    model.addConstr(gp.quicksum(b_in[i, j] for j in range(4)), '<=', 3, name="C8")

# Constraints for ensuring the aircraft doesn't break laws of physics
C8 = {}
C9 = {}
for i in range(number_of_time_steps - 1):
    C8[i, 0] = model.addLConstr(x[i+1] - x[i], '<=', v_max_x, name=f"C8_{i},0")
    C8[i, 1] = model.addLConstr(x[i + 1] - x[i], '>=', -v_max_x, name=f"C8_{i},1")
    C9[i, 0] = model.addLConstr(y[i+ 1] - y[i], '<=', v_max_y, name=f"C9_{i},0")
    C9[i, 1] = model.addLConstr(y[i + 1] - y[i], '>=', -v_max_y, name=f"C9_{i},1")


model.setObjective(gp.quicksum(b_goal[i] * i for i in range(number_of_time_steps)), gp.GRB.MINIMIZE)

model.update()

model.write("models/model.lp")

model.optimize()

x_result = []
y_result = []

for i in range(number_of_time_steps):
    x_result.append(x[i].x)
    y_result.append(y[i].x)

plt.plot(x_result, y_result, marker = '.')

for obstacle in list_of_obstacles:
    x = [obstacle.x_upper_right, obstacle.x_upper_right, 
         obstacle.x_lower_left, obstacle.x_lower_left, obstacle.x_upper_right]
    y = [obstacle.y_upper_right, obstacle.y_lower_left, 
         obstacle.y_lower_left, obstacle.y_upper_right, obstacle.y_upper_right]
    plt.plot(x, y)

plt.show()