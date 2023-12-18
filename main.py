import gurobipy as gp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import numpy as np
import networkx as nx


def receding_horizon(x_init, v_init, list_of_obstacles, x_goal, v_max=10, u_max=0.1, r_plan=150, number_of_steps=250,
                     NORMALS=5, DIMENSION=2, map_bound=np.array([[0, 0], [600, 600]])):
    # Limits of the map, width is 600px, height is 400px

    # Create MILP model.
    model = gp.Model("Path Planning")

    # Model constants
    R = 1e6  # Big number.
    # NORMALS = 5  # Number of normals used for vector magnitude calcualtion.
    # DIMENSION = 2  # Number of dimensions.

    # Define obstacles.
    # list_of_obstacles.append(np.array([[150, 200], [200, 410]]))    # Obstacle bounds (dx2 array with [lower_left, upper_right])
    # list_of_obstacles.append(np.array([[10, -10], [30, 250]]))

    # Initial conditions of the vehicle.
    # x_init = [0, 0]  # Initial position of the vehicle.
    # v_init = [100, 0]  # Initial velocity of the vehicle.

    # Final conditions of the vehicle at the goal.
    # x_goal = (320, 0)  # Goal position

    # Equivalent number of time steps, L.
    number_of_time_steps = number_of_steps

    # Limit on parameters
    # map_bound = np.array([[0, 0], [600, 600]])  # Map bounds (dx2 array with [lower_left, upper_right])
    # v_max = 101  # Maximum velocity (scalar)
    # u_max = 0  # Maximum input (scalar)

    # Define variable dictionaries.
    x = {}  # Decision variables for the position of the rover.
    u = {}  # Decision variables for the inputs of the rover.
    b_goal = {}  # Boolean indicating whether a node reaches the goal.
    b_in = {}  # Boolean used for checking collision with obstacles.
    x_goal_range = {}  # Range of the goal position.

    # ================ DEFINE MODEL VARIABLES =========================

    # Define the variables for each time step.
    for i in range(number_of_time_steps):

        for d in range(DIMENSION):
            # Position variables for the vehicle position, limited by the map bounds.
            x[i, d] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"X[time={i},dim={d}]", lb=map_bound[0, d],
                                   ub=map_bound[1, d])

            # Input variables, limited by maximum input.
            u[i, d] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"U[time={i},dim={d}]", lb=-u_max - R, ub=u_max + R)

            # Binary for obstacle detection.
            for k in range(len(list_of_obstacles)):
                # Binaries for detecting obstacle collision.
                b_in[i, d, k, 0] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_in[time={i},dim={d},obs={k},bound={0}]")
                b_in[i, d, k, 1] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_in[time={i},dim={d},obs={k},bound={1}]")

        # Binary indicating first node that reaches goal.
        b_goal[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_goal_{i}")

    # x_goal_range[0] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"B_goal_range", lb=-v_max, ub=v_max)
    # x_goal_range[1] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"B_goal_range", lb=-v_max, ub=v_max)

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

    for i in range(number_of_time_steps - 1):
        for d in range(DIMENSION):
            CONST_X_GOAL[i, d, 0] = model.addLConstr(x[i, d] - x_goal[d], '<=', R * (1 - b_goal[i]),
                                                     name=f"CONST_X_GOAL[time={i},dim={d},const=0]")
            CONST_X_GOAL[i, d, 1] = model.addLConstr(x[i, d] - x_goal[d], '>=', -R * (1 - b_goal[i]),
                                                     name=f"CONST_X_GOAL[time={i},dim={d},const=1]")

    # Constraint for ensuring exactly one node reaches the goal.
    CONST_REACH_GOAL = model.addLConstr(gp.quicksum(b_goal[i] for i in range(number_of_time_steps)), '=', 1,
                                        name="CONST_REACH_GOAL")

    # Constraints for ensuring the robot does not collide with the obstacles.
    CONST_OBSTACLE = {}
    CONST_OBSTACLE_DETECTION = {}

    for i in range(number_of_time_steps):

        for k, obstacle in enumerate(list_of_obstacles):

            for d in range(DIMENSION):
                CONST_OBSTACLE[i, d, k, 0] = model.addLConstr(x[i, d], '<=', obstacle[0, d] + R * b_in[i, d, k, 0],
                                                              name=f"CONST_OBSTACLE[time={i},dim={d},obs={k},bound={0}]")
                CONST_OBSTACLE[i, d, k, 1] = model.addLConstr(x[i, d], '>=', obstacle[1, d] - R * b_in[i, d, k, 1],
                                                              name=f"CONST_OBSTACLE[time={i},dim={d},obs={k},bound={1}]")

            CONST_OBSTACLE_DETECTION[i, k] = model.addLConstr(
                gp.quicksum(b_in[i, d, k, 0] + b_in[i, d, k, 1] for d in range(DIMENSION)), '<=', 3,
                name=f"CONST_OBSTACLE_DETECTION[time={i},obs={k}]")

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

            # CONST_V_MAX[i, n] = model.addLConstr( (x[i + 1, 0] - x[i, 0]) * 1 + (x[i + 1, 1] - x[i, 1]) * 0, '<=',
            # v_max, name=f"COST_V_MAX[time={i},normal={n}]")

            # Input constrain
            CONST_U_MAX[i, n] = model.addLConstr(
                (u[i, 0]) * np.cos(2 * np.pi / NORMALS * n) + (u[i, 1]) * np.sin(
                    2 * np.pi / NORMALS * n), '<=', u_max + b_goal[i + 1] * R * 0,
                name=f"CONST_U_MAX[time={i},normals={n}]")

            # CONST_U_MAX[i, n] = model.addLConstr( (u[i + 1, 0] - u[i, 0]) * 1 + (u[i + 1, 1] - u[i, 1]) * 0, '<=',
            # u_max, name=f"ONST_U_MAX[time={i},normals={n}]")

    # Relation input, velocity and position next point.
    CONST_POS = {}

    # Correlate input and the velocity
    for i in range(number_of_time_steps - 2):
        for d in range(DIMENSION):
            CONST_POS[i, d] = model.addLConstr(x[i + 2, d], '=', 2 * x[i + 1, d] - x[i, d] + u[i + 1, d],
                                               name=f"CONST_POS[time={i},dim={d}]")

    OBJECTIVE = model.setObjective(gp.quicksum(b_goal[i] * i for i in range(number_of_time_steps)), gp.GRB.MINIMIZE)

    model.update()

    model.write("models/model.lp")

    model.optimize()

    # Construct list with vehicle location to be plotted
    x_plot = []
    y_plot = []
    objective_value = model.getObjective().getValue()

    for i in range(number_of_time_steps):
        x_plot.append(x[i, 0].X)
        y_plot.append(x[i, 1].X)
        # Stop plot once goal is reached.
        if b_goal[i].X == 1:
            break

    return np.array([x_plot, y_plot]), objective_value

    # ============== PLOT RESULTS =================================


def plot(path, list_of_obstacles, map_bound=np.array([[0, 0], [600, 600]])):
    # Define figure and axis.
    fig, ax = plt.subplots()

    # Set limits
    ax.set_xlim(map_bound[0, 0], map_bound[1, 0])
    ax.set_ylim(map_bound[0, 1], map_bound[1, 1])

    # Plot vehicle location
    ax.plot(path[0], path[1], marker=".", color='red', label="Path")

    # Plot obstacles
    for obstacle in list_of_obstacles:
        origin = obstacle[0]
        delta = obstacle[1] - obstacle[0]
        width = delta[0]
        height = delta[1]

        ax.add_patch(Rectangle(origin, width, height, color='dimgrey'))

    # display plot
    plt.show()


# ================== COLISION DETECTION ==========================
def collision_detection(x, list_of_obstacles):
    for i in range(len(list_of_obstacles)):
        # Check if point is in obstacle
        if list_of_obstacles[i][0, 0] <= x[0] <= list_of_obstacles[i][1, 0] and list_of_obstacles[i][0, 1] <= x[1] <= \
                list_of_obstacles[i][1, 1]:
            return True
    return False


# ================== GENERATE NODES ======================
def generate_map(list_of_obstacles, map_bounds, x_goal, skip_factor):
    connection_list = nx.Graph([])
    for i in range(map_bounds[0, 0], map_bounds[1, 0], skip_factor):
        for j in range(map_bounds[0, 1], map_bounds[1, 1], skip_factor):
            if collision_detection([i, j], list_of_obstacles):
                continue
            else:
                connection_list.add_node(f"X:{i}, Y:{j}", pos=(i, j))
                for k in [-1*skip_factor, 0, 1*skip_factor]:
                    for l in [-1*skip_factor, 0, 1*skip_factor]:
                        if collision_detection([i + k, j + l], list_of_obstacles):
                            continue
                        else:
                            connection_list.add_edge(f"X:{i}, Y:{j}", f"X:{i+k}, Y:{j+l}", weight=math.sqrt(k ** 2 + l ** 2))
    return connection_list


# ================== A* ALGORITHM ======================

def cost_cal(graph, x_goal):
    cost = nx.single_source_bellman_ford_path_length(graph, f"X:{x_goal[0]}, Y:{x_goal[1]}")

    return dict(cost)

# ================== MAIN =======================================

if __name__ == '__main__':
    # Model constants
    R = 1e6  # Big number.
    # NORMALS = 5  # Number of normals used for vector magnitude calcualtion.
    # DIMENSION = 2  # Number of dimensions.
    list_of_obstacles = []
    # Define obstacles.
    list_of_obstacles.append(
        np.array([[150, 200], [200, 410]]))  # Obstacle bounds (dx2 array with [lower_left, upper_right])
    list_of_obstacles.append(np.array([[10, -10], [30, 250]]))
    list_of_obstacles.append(np.array([[50, 50], [80, 400]]))

    # Initial conditions of the vehicle.
    x_init = [0, 0]  # Initial position of the vehicle.
    v_init = [0, 0]  # Initial velocity of the vehicle.

    # Final conditions of the vehicle at the goal.
    x_goal = (300, 300)  # Goal position

    # Limit on parameters
    map_bound = np.array([[0, 0], [400, 350]])  # Map bounds (dx2 array with [lower_left, upper_right])
    v_max = 5  # Maximum velocity (scalar)
    u_max = 5  # Maximum input (scalar)

    r_plan = 150
    number_of_steps = 300
    NORMALS = 16
    DIMENSION = 2

    point_graph = generate_map(list_of_obstacles, map_bound, x_goal, 1)
    cost_array = cost_cal(point_graph, x_goal)
    print(cost_array['X:0, Y:0'])
    # plt.show()

    # path, objective_resutl = receding_horizon(x_init, v_init, list_of_obstacles, x_goal, v_max, u_max, r_plan,
    #                                           number_of_steps, NORMALS, DIMENSION, map_bound)
    #
    # plot(path, list_of_obstacles, map_bound)
