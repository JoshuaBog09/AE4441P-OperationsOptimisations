import gurobipy as gp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import re
import numpy as np
import networkx as nx


def receding_horizon(x_init, v_init, list_of_obstacles, x_goal, distance_map, v_max=10, u_max=0.1, r_plan=150,
                     number_of_steps=250,
                     NORMALS=5, DIMENSION=2, map_bound=np.array([[0, 0], [600, 600]]), goal_constrains=False):
    # Limits of the map, width is 600px, height is 400px

    # Create MILP model.
    model = gp.Model("Path Planning")

    # Model constants
    R = 1e6  # Big number.
    # NORMALS = 5  # Number of normals used for vector magnitude calcualtion.
    # DIMENSION = 2  # Number of dimensions.

    # Define obstacles. list_of_obstacles.append(np.array([[150, 200], [200, 410]]))    # Obstacle bounds (dx2 array
    # with [lower_left, upper_right]) list_of_obstacles.append(np.array([[10, -10], [30, 250]]))

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
    b_reach = {}  # Boolean indicating whether the goal is reachable.
    b_in = {}  # Boolean used for checking collision with obstacles.
    normal_direction_distance = {}  # Normal direction distance.
    goal_distance = {}  # Distance to goal for each node.
    x_goal_range = {}  # Range of the goal position.
    interpolation_points = {}
    interpolation_points_in = {}

    # ================ DEFINE MODEL VARIABLES =========================
    # Binary deciding if goal is reachable
    b_reach[0] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_reach")

    # Define the variables for each time step.
    for i in range(number_of_time_steps):

        for d in range(DIMENSION):
            # Position variables for the vehicle position, limited by the map bounds.
            x[i, d] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"X[time={i},dim={d}]", lb=map_bound[0, d],
                                   ub=map_bound[1, d])

            # Input variables, limited by maximum input.
            u[i, d] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"U[time={i},dim={d}]", lb=-u_max - R, ub=u_max + R)

            # Distance to goal for each node
            goal_distance[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"goal_distance[time={i}]", lb=0,
                                            ub=map_bound[1, 0] + map_bound[1, 1])

            for n in range(NORMALS):
                # Normal direction distance
                normal_direction_distance[i, n] = model.addVar(vtype=gp.GRB.CONTINUOUS,
                                                               name=f"normal_direction_distance[time={i},normal={n}]",
                                                               lb=-10000, ub=10000)

            # Binary for obstacle detection.
            for k in range(len(list_of_obstacles)):
                # Binaries for detecting obstacle collision.
                b_in[i, d, k, 0] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_in[time={i},dim={d},obs={k},bound={0}]")
                b_in[i, d, k, 1] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_in[time={i},dim={d},obs={k},bound={1}]")

        # Binary indicating first node that reaches goal.
        b_goal[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_goal_{i}")

    # x_goal_range[0] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"B_goal_range", lb=-v_max, ub=v_max)
    # x_goal_range[1] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"B_goal_range", lb=-v_max, ub=v_max)

    # Interpolation squares
    
    for i in range(map_bound[0, 0], map_bound[1, 0], skip_factor):
        for j in range(map_bound[0, 1], map_bound[1, 1], skip_factor):
            
            interpolation_points[i,j] = model.addVar(vtype=gp.GRB.BINARY, name=f"Interpolation_point[x_low={i},y_low={j}]")

            for k in range(4):

                interpolation_points_in[i,j,k] = model.addVar(vtype=gp.GRB.BINARY, name=f"Ip_in[x_low={i},y_low={j},bound={k}]")

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
    CONST_REACH_GOAL = model.addLConstr(gp.quicksum(b_goal[i] for i in range(number_of_time_steps)), '=', b_reach[0],
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

    # TODO: Figure out how "interpolate" the distance map. One idea is to create new collision rhomboids around very
    #  point and when the an X is in that collision box, then it has that collision box's distance value

    CONST_IPSQUARES = {}
    CONST_IPSQUARE1 = {}
    CONST_IPSQUARE2 = {}
    
    trailing_x = x[number_of_time_steps-1, 0]
    trailing_y = x[number_of_time_steps-1, 1]

    for i in range(map_bound[0, 0], map_bound[1, 0], skip_factor):
        for j in range(map_bound[0, 1], map_bound[1, 1], skip_factor):
            trailing_x = 10
            trailing_y = 10

            CONST_IPSQUARES[i,j,0] = model.addLConstr(trailing_x, ">=", (i - skip_factor/2) - R * interpolation_points_in[i,j,0],
                                                      name=f"CONST_IPSQUARE_DETECTION[x_low={i},y_low={j},bound={0}]")
            CONST_IPSQUARES[i,j,1] = model.addLConstr(trailing_x, "<", (i + skip_factor/2) + R * interpolation_points_in[i,j,1],
                                                      name=f"CONST_IPSQUARE_DETECTION[x_low={i},y_low={j},bound={1}]")
            CONST_IPSQUARES[i,j,2] = model.addLConstr(trailing_y, ">=", (j - skip_factor/2) - R * interpolation_points_in[i,j,2],
                                                      name=f"CONST_IPSQUARE_DETECTION[x_low={i},y_low={j},bound={2}]")
            CONST_IPSQUARES[i,j,3] = model.addLConstr(trailing_y, "<", (j + skip_factor/2) + R * interpolation_points_in[i,j,3],
                                                      name=f"CONST_IPSQUARE_DETECTION[x_low={i},y_low={j},bound={3}]")

            CONST_IPSQUARE1[i,j] = model.addLConstr(gp.quicksum(interpolation_points_in[i,j,k] for k in range(4)) + interpolation_points[i,j], ">=",
                                                   1,
                                                   name=f"CONST_IPSQUARE_DETECTION[x_low={i},y_low={j}]")

    CONST_IPSQUARE2[0] = model.addLConstr(gp.quicksum(interpolation_points[i,j] for i in range(map_bound[0, 0], map_bound[1, 0], skip_factor) for j in range(map_bound[0, 1], map_bound[1, 1], skip_factor)),
                                         "=",1)

    # Constraints for ensuring the aircraft doesn't break laws of physics
    CONST_V_MAX = {}
    CONST_U_MAX = {}
    CONST_GOAL_DISTANCE = {}

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
    CONST_NORMAL_DISTANCE = {}

    for i in range(number_of_time_steps):
        for n in range(NORMALS):
            CONST_NORMAL_DISTANCE[i, n] = model.addLConstr((x_goal[0] - x[i, 0]) * np.cos(2 * np.pi / NORMALS * n) + (
                    x_goal[1] - x[i, 1]) * np.sin(2 * np.pi / NORMALS * n), '=', normal_direction_distance[i, n],
                                                           name=f"CONST_NORMAL_DISTANCE[time={i},normal={n}]")
        # TODO: maybe remove forbidden magic ("gp.max_")
        CONST_GOAL_DISTANCE[i] = model.addConstr(
            goal_distance[i] == gp.max_(normal_direction_distance[i, n] for n in range(NORMALS)),
            name=f"CONST_GOAL_DISTANCE[time={i}]")

    # Correlate input and the velocity
    for i in range(number_of_time_steps - 2):
        for d in range(DIMENSION):
            CONST_POS[i, d] = model.addLConstr(x[i + 2, d], '=', 2 * x[i + 1, d] - x[i, d] + u[i + 1, d],
                                               name=f"CONST_POS[time={i},dim={d}]")

    # **** for gurobi
    object_var = {}
    object_constraint = {}
    object_var[0] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"object_var")

    # TODO: make sure this constraint is allowed
    model.addConstr(object_var[0] == gp.min_(goal_distance[i] for i in range(number_of_time_steps)),
                    name=f"object_constraint")
    # object_constraint[0] = model.addConstr(object_var[0] == gp.min_(x_goal[0] - x[i, 0] + x_goal[1] - x[i,
    # 1] for i in range(number_of_time_steps)), name=f"object_constraint") OBJECTIVE = model.setObjective(
    # gp.quicksum(b_goal[i] * i for i in range(number_of_time_steps)), gp.GRB.MINIMIZE)
    OBJECTIVE = model.setObjective(object_var[0] - number_of_time_steps * b_reach[0] + gp.quicksum(b_goal[i] * i for i in range(number_of_time_steps)), gp.GRB.MINIMIZE)
    model.update()

    model.write("models/model.lp")

    model.optimize()

    # for i in range(number_of_time_steps):
    #     print(f"Position for point {i} is {x[i, 0].X}, {x[i, 1].X}")
    #     print(f"Distance for point {i} is {goal_distance[i].X}")
    #     print(f"True distance for point {i} is {max((x_goal[0] - x[i, 0].X) * np.cos(2 * np.pi / NORMALS * n) + (x_goal[1] - x[i, 1].X) * np.sin(2 * np.pi / NORMALS * n) for n in range(NORMALS))}")

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
    
    for i in range(map_bound[0, 0], map_bound[1, 0], skip_factor):
        for j in range(map_bound[0, 1], map_bound[1, 1], skip_factor):

            if interpolation_points[i,j].X == 1:
                print(i,j)

    print(x[number_of_time_steps-1, 0].X, x[number_of_time_steps-1, 1].X)


    return np.array([x_plot, y_plot]), objective_value


# ============== PLOT RESULTS =================================


def plot(path, x_pos,obstacle_list, map_limit=np.array([[0, 0], [600, 600]])):
    # Define figure and axis.
    fig, ax = plt.subplots()

    # Set limits
    ax.set_xlim(map_limit[0, 0], map_limit[1, 0])
    ax.set_ylim(map_limit[0, 1], map_limit[1, 1])

    # Plot vehicle location
    ax.plot(path[0], path[1], marker=".", color='red', label="Path")
    ax.plot(x_pos[0], x_pos[1], marker="*", color='green', label="Goal")

    # Plot obstacles
    for obstacle in obstacle_list:
        origin = obstacle[0]
        delta = obstacle[1] - obstacle[0]
        width = delta[0]
        height = delta[1]

        ax.add_patch(Rectangle(origin, width, height, color='dimgrey'))

    # display plot
    plt.show()


# ================== PLOT DISTANCE TO GOAL HEAT MAP ======================
def plot_distance_to_goal_heat_map(obstacle_list, distance_graph, skip_factor,
                                   map_limit=np.array([[0, 0], [600, 600]])):
    distance_list = np.zeros(shape=(map_limit[1, 0] // skip_factor, map_limit[1, 1] // skip_factor))
    for i in range(map_limit[0, 0], map_limit[1, 0], skip_factor):
        for j in range(map_limit[0, 1], map_limit[1, 1], skip_factor):
            if collision_detection([i, j], obstacle_list):
                distance_list[i, j] = 0
            else:
                distance_list[i, j] = distance_graph[f"X:{i}, Y:{j}"]

    # Define figure and axis.
    fig, ax = plt.subplots()

    # Set limits
    ax.set_xlim(map_limit[0, 0], map_limit[1, 0])
    ax.set_ylim(map_limit[0, 1], map_limit[1, 1])

    distance_list = np.flip(distance_list, 0)
    distance_list = np.rot90(distance_list, -1)

    im = ax.imshow(distance_list, origin='lower', cmap='plasma', interpolation='none')

    plt.colorbar(im, ax=ax)

    # Plot obstacles
    for obstacle in obstacle_list:
        origin = obstacle[0]
        delta = obstacle[1] - obstacle[0]
        width = delta[0]
        height = delta[1]

        ax.add_patch(Rectangle(origin, width, height, color='dimgrey'))

    plt.show()


# ================== COLLISION DETECTION ==========================
def collision_detection(x, obstacle_list):
    for i in range(len(obstacle_list)):
        # Check if point is in obstacle
        if obstacle_list[i][0, 0] <= x[0] <= obstacle_list[i][1, 0] and obstacle_list[i][0, 1] <= x[1] <= \
                obstacle_list[i][1, 1]:
            return True
    return False


# ================== GENERATE NODES ======================
def generate_map(obstacle_list, map_bounds, skip_factor):
    connection_list = nx.Graph([])
    for i in range(map_bounds[0, 0], map_bounds[1, 0], skip_factor):
        for j in range(map_bounds[0, 1], map_bounds[1, 1], skip_factor):
            if collision_detection([i, j], obstacle_list):
                continue
            else:
                connection_list.add_node(f"X:{i}, Y:{j}", pos=(i, j))
                for k in [-1 * skip_factor, 0, 1 * skip_factor]:
                    for l in [-1 * skip_factor, 0, 1 * skip_factor]:
                        if collision_detection([i + k, j + l], obstacle_list):
                            continue
                        else:
                            connection_list.add_edge(f"X:{i}, Y:{j}", f"X:{i + k}, Y:{j + l}",
                                                     weight=math.sqrt(k ** 2 + l ** 2))
    return connection_list


# ================== A* ALGORITHM ======================

def cost_cal(graph, goal_pos):
    cost = nx.single_source_bellman_ford_path_length(graph, f"X:{goal_pos[0]}, Y:{goal_pos[1]}")

    return dict(cost)


# ================== MAIN =======================================

if __name__ == '__main__':
    # Model constants
    R = 1e6  # Big number.
    # NORMALS = 5  # Number of normals used for vector magnitude calculation.
    # DIMENSION = 2  # Number of dimensions.
    list_of_obstacles = []
    # Define obstacles.
    # list_of_obstacles.append(
    #     np.array([[150, 200], [200, 410]]))  # Obstacle bounds (dx2 array with [lower_left, upper_right])
    # list_of_obstacles.append(np.array([[10, -10], [30, 250]]))
    # list_of_obstacles.append(np.array([[50, 50], [80, 400]]))
    # list_of_obstacles.append(np.array([[250, 250], [420, 260]]))
    # list_of_obstacles.append(np.array([[250, 250], [260, 335]]))
    # list_of_obstacles.append(np.array([[270, 210], [420, 220]]))
    # list_of_obstacles.append(np.array([[390, 210], [420, 260]]))

    # Initial conditions of the vehicle.
    x_init = [1, 1]  # Initial position of the vehicle.
    v_init = [0, 0]  # Initial velocity of the vehicle.

    # Final conditions of the vehicle at the goal.
    x_goal = (39, 34)  # Goal position

    # Limit on parameters
    map_bound = np.array([[0, 0], [40, 35]])  # Map bounds (dx2 array with [lower_left, upper_right])
    v_max = 5  # Maximum velocity (scalar)
    u_max = 1  # Maximum input (scalar)

    r_plan = 150
    number_of_steps = 4
    NORMALS = 16
    DIMENSION = 2
    skip_factor = 1  # this is a little broken, keep it at 1 for now

    point_graph = generate_map(list_of_obstacles, map_bound, skip_factor)
    cost_array = cost_cal(point_graph, x_goal)
    # plt.show()
    plot_distance_to_goal_heat_map(list_of_obstacles, cost_array, skip_factor, map_bound)
    path, objective_result = receding_horizon(x_init, v_init, list_of_obstacles, x_goal, cost_array, v_max, u_max,
                                              r_plan,
                                              number_of_steps, NORMALS, DIMENSION, map_bound, False)

    plot(path, x_goal, list_of_obstacles, map_bound)
