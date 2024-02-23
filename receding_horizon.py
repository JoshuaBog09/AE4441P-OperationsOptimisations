'''
@File           :
@Date           :   dd-mm-yyyy
@Aauthor        :   Justin Dubois
@Contact        :   j.p.g.dubois@student.tudelft.nl
@Version        :   1.0
@License        :
@Description    :
'''

import gurobipy as gp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
import re
import numpy as np
import networkx as nx
import pickle

class Scene:

    def __init__(self, map_bounds, obstacles, goal):
        self.map_bounds   = map_bounds
        self.obstacles    = obstacles
        self.goal         = goal
        self.gradient_map = self.a_star()

    def collision_detection(self, x):
        '''
        Check if a point in the map is in an obstacle.
        '''
        for i in range(len(self.obstacles)):
            # Check if point is in obstacle
            if self.obstacles[i][0, 0] <= x[0] <= self.obstacles[i][1, 0] and self.obstacles[i][0, 1] <= x[1] <= \
                    self.obstacles[i][1, 1]:
                return True
        return False

    def a_star(self):
        graph = nx.Graph([])

        for i in range(self.map_bounds[0, 0], self.map_bounds[1, 0]):
            for j in range(self.map_bounds[0, 1], self.map_bounds[1, 1]):
                if self.collision_detection([i, j]):
                    continue
                else:
                    graph.add_node(f"X:{i}, Y:{j}", pos=(i, j))
                    for k in [-1, 0, 1]:
                        for l in [-1, 0, 1]:
                            if self.collision_detection([i + k, j + l]):
                                continue
                            else:
                                graph.add_edge(f"X:{i}, Y:{j}", f"X:{i + k}, Y:{j + l}",
                                                         weight=math.sqrt(k ** 2 + l ** 2))
        cost = dict(nx.single_source_bellman_ford_path_length(graph, f"X:{self.goal[0]}, Y:{self.goal[1]}"))

        for i in range(self.map_bounds[0, 0], self.map_bounds[1, 0]):
            for j in range(self.map_bounds[0, 1], self.map_bounds[1, 1]):
                if f"X:{i}, Y:{j}" not in cost:
                    cost[f"X:{i}, Y:{j}"] = 1e8
        return cost

    def show_scene(self):
        distance_list = np.zeros(shape=(self.map_bounds[1, 0], self.map_bounds[1, 1]))
        for i in range(self.map_bounds[0, 0], self.map_bounds[1, 0]):
            for j in range(self.map_bounds[0, 1], self.map_bounds[1, 1]):
                if self.collision_detection([i, j]):
                    distance_list[i, j] = 0
                else:
                    distance_list[i, j] = self.gradient_map[f"X:{i}, Y:{j}"]

        # Define figure and axis.
        fig, ax = plt.subplots()

        # Set limits
        ax.set_xlim(self.map_bounds[0, 0], self.map_bounds[1, 0])
        ax.set_ylim(self.map_bounds[0, 1], self.map_bounds[1, 1])

        distance_list = np.flip(distance_list, 0)
        distance_list = np.rot90(distance_list, -1)

        im = ax.imshow(distance_list, origin='lower', cmap='plasma', interpolation='none')

        plt.colorbar(im, ax=ax)

        # Plot obstacles
        for obstacle in self.obstacles:
            origin = obstacle[0]
            delta = obstacle[1] - obstacle[0]
            width = delta[0]
            height = delta[1]

            ax.add_patch(Rectangle(origin, width, height, color='dimgrey'))

        ax.plot(self.goal[0], self.goal[1], marker="X", color = "r", label = "goal")
        plt.show()

class Config:

    def __init__(self, normals, dimension, steps, big_m = 1e6):
        self.normals   = normals
        self.dimension = dimension
        self.steps = steps
        self.big_m = big_m

class Vehicle:

    def __init__(self, v_max, u_max, x_init, v_init):
        self.v_max = v_max
        self.u_max = u_max
        self.x_init = x_init
        self.v_init = v_init

def make_model(scene, config, vehicle):

    # Create MILP model.
    model = gp.Model("Path Planning")

    # Model constants
    R =  config.big_m # Big number.
    NORMALS = config.normals  # Number of normals used for vector magnitude calculation.
    DIMENSION = config.dimension  # Number of dimensions.

    # Define obstacles
    list_of_obstacles = scene.obstacles


    # Initial conditions of the vehicle.
    x_init = vehicle.x_init  # Initial position of the vehicle.
    v_init = vehicle.v_init  # Initial velocity of the vehicle.

    # Final conditions of the vehicle at the goal.
    x_goal = scene.goal  # Goal position

    # Equivalent number of time steps, L.
    number_of_time_steps = config.steps

    distance_map_dict = scene.gradient_map

    # Limit on parameters
    map_bound = scene.map_bounds  # Map bounds (dx2 array with [lower_left, upper_right])
    v_max = vehicle.v_max  # Maximum velocity (scalar)
    u_max = vehicle.u_max  # Maximum input (scalar)

    # Define variable dictionaries.
    x = {}  # Decision variables for the position of the rover.
    u = {}  # Decision variables for the inputs of the rover.
    b_goal = {}  # Boolean indicating whether a node reaches the goal.
    b_reach = {}  # Boolean indicating whether the goal is reachable.
    b_active_dmap_node = {}  # Boolean indicating whether a node is active in the distance map.
    b_in = {}  # Boolean used for checking collision with obstacles.
    # normal_direction_distance = {}  # Normal direction distance.
    goal_distance = {}  # Distance to goal for each node.
    x_goal_range = {}  # Range of the goal position.
    interpolation_points = {}

    # ================ DEFINE DISTANCE MAP =========================
    # Convert the distance map dictionary to a list of lists.
    distance_map = []

    for key, value in distance_map_dict.items():
        distance_map.append([[int(re.findall(r'\b\d+\b', key)[0]), int(re.findall(r'\b\d+\b', key)[1])], value])

    # ================ DEFINE MODEL VARIABLES =========================
    # Binary deciding if goal is reachable
    b_reach[0] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_reach")

    # Define the variables for each time step.
    for i in range(number_of_time_steps):

        goal_distance[i] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"goal_distance[time={i}]", lb=0,
                                        ub=map_bound[1, 0] + map_bound[1, 1])

        for d in range(DIMENSION):
            # Position variables for the vehicle position, limited by the map bounds.
            x[i, d] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"X[time={i},dim={d}]", lb=map_bound[0, d],
                                   ub=map_bound[1, d])

            # Input variables, limited by maximum input.
            u[i, d] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"U[time={i},dim={d}]", lb=-u_max - R, ub=u_max + R)

            # Distance to goal for each node

            # for n in range(NORMALS):
            # Normal direction distance
            # normal_direction_distance[i, n] = model.addVar(vtype=gp.GRB.CONTINUOUS,
            #                                                name=f"normal_direction_distance[time={i},normal={n}]",
            #                                                lb=-10000, ub=10000)

            # Binary for obstacle detection.
            for k in range(len(list_of_obstacles)):
                # Binaries for detecting obstacle collision.
                b_in[i, d, k, 0] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_in[time={i},dim={d},obs={k},bound={0}]")
                b_in[i, d, k, 1] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_in[time={i},dim={d},obs={k},bound={1}]")

        # Binary indicating first node that reaches goal.
        b_goal[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_goal_{i}")

    for i, node in enumerate(distance_map):
        b_active_dmap_node[i] = model.addVar(vtype=gp.GRB.BINARY, name=f"B_active_dmap_node_{i}")

    # x_goal_range[0] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"B_goal_range", lb=-v_max, ub=v_max)
    # x_goal_range[1] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"B_goal_range", lb=-v_max, ub=v_max)

    # Interpolation squares

    for i in range(map_bound[0, 0], map_bound[1, 0], skip_factor):
        for j in range(map_bound[0, 1], map_bound[1, 1], skip_factor):
            interpolation_points[i, j] = model.addVar(vtype=gp.GRB.BINARY,
                                                      name=f"Interpolation_point[x_low={i},y_low={j}]")

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

    CONST_IPSQUARES = {}
    CONST_IPSQUARE1 = {}
    CONST_IPSQUARE2 = {}

    trailing_x = x[number_of_time_steps - 1, 0]
    trailing_y = x[number_of_time_steps - 1, 1]

    for i in range(map_bound[0, 0], map_bound[1, 0], skip_factor):
        for j in range(map_bound[0, 1], map_bound[1, 1], skip_factor):
            CONST_IPSQUARES[i, j, 0] = model.addLConstr(trailing_x, ">=",
                                                        (i - skip_factor / 2) - R * (1 - interpolation_points[i, j]),
                                                        name=f"CONST_IPSQUARE_DETECTION[x_low={i},y_low={j},bound={0}]")
            CONST_IPSQUARES[i, j, 1] = model.addLConstr(trailing_x, "<",
                                                        (i + skip_factor / 2) + R * (1 - interpolation_points[i, j]),
                                                        name=f"CONST_IPSQUARE_DETECTION[x_low={i},y_low={j},bound={1}]")
            CONST_IPSQUARES[i, j, 2] = model.addLConstr(trailing_y, ">=",
                                                        (j - skip_factor / 2) - R * (1 - interpolation_points[i, j]),
                                                        name=f"CONST_IPSQUARE_DETECTION[x_low={i},y_low={j},bound={2}]")
            CONST_IPSQUARES[i, j, 3] = model.addLConstr(trailing_y, "<",
                                                        (j + skip_factor / 2) + R * (1 - interpolation_points[i, j]),
                                                        name=f"CONST_IPSQUARE_DETECTION[x_low={i},y_low={j},bound={3}]")

    CONST_IPSQUARE2[0] = model.addLConstr(gp.quicksum(
        interpolation_points[i, j] for i in range(map_bound[0, 0], map_bound[1, 0], skip_factor) for j in
        range(map_bound[0, 1], map_bound[1, 1], skip_factor)),
        "=", 1)

    # Constrains for the binary values that will tell us which node is active in the distance map.
    # CONST_DMAP = {}
    # for i in range(number_of_time_steps):
    #     for j, node in enumerate(distance_map):
    #         for d in range(DIMENSION):
    #             continue

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
                    2 * np.pi / NORMALS * n), '<=', u_max,
                name=f"CONST_U_MAX[time={i},normals={n}]")

            # CONST_U_MAX[i, n] = model.addLConstr( (u[i + 1, 0] - u[i, 0]) * 1 + (u[i + 1, 1] - u[i, 1]) * 0, '<=',
            # u_max, name=f"ONST_U_MAX[time={i},normals={n}]")

    # Relation input, velocity and position next point.
    # CONST_POS = {}
    # CONST_NORMAL_DISTANCE = {}

    # for i in range(number_of_time_steps):
    #     for n in range(NORMALS):
    #         CONST_NORMAL_DISTANCE[i, n] = model.addLConstr((x_goal[0] - x[i, 0]) * np.cos(2 * np.pi / NORMALS * n) + (
    #                 x_goal[1] - x[i, 1]) * np.sin(2 * np.pi / NORMALS * n), '=', normal_direction_distance[i, n],
    #                                                        name=f"CONST_NORMAL_DISTANCE[time={i},normal={n}]")

    # Correlate input and the velocity
    # for i in range(number_of_time_steps - 2):
    #     for d in range(DIMENSION):
    #         CONST_POS[i, d] = model.addLConstr(x[i + 2, d], '=', 2 * x[i + 1, d] - x[i, d] + u[i + 1, d],
    #                                            name=f"CONST_POS[time={i},dim={d}]")

    object_var = {}
    object_constraint = {}
    object_var[0] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"object_var")

    OBJECTIVE = model.setObjective(gp.quicksum(interpolation_points[i, j] * distance_map_dict[f"X:{i}, Y:{j}"] for i in
                                               range(map_bound[0, 0], map_bound[1, 0], skip_factor) for j in
                                               range(map_bound[0, 1], map_bound[1, 1],
                                                     skip_factor)) - number_of_time_steps * b_reach[0] + gp.quicksum(
        b_goal[i] * i for i in range(number_of_time_steps)), gp.GRB.MINIMIZE)
    model.update()

    return model

    # model.write("models/model.lp")


    # for i in range(number_of_time_steps):
    #     print(f"Position for point {i} is {x[i, 0].X}, {x[i, 1].X}")
    #     print(f"Distance for point {i} is {goal_distance[i].X}")
    #     print(f"True distance for point {i} is {max((x_goal[0] - x[i, 0].X) * np.cos(2 * np.pi / NORMALS * n) + (x_goal[1] - x[i, 1].X) * np.sin(2 * np.pi / NORMALS * n) for n in range(NORMALS))}")



if __name__ == "__main__":
    list_of_obstacles = []
    # Define obstacles.
    list_of_obstacles.append(np.array([[150, 200], [200, 410]]))  # Obstacle bounds (dx2 array with [lower_left, upper_right])
    list_of_obstacles.append(np.array([[10, -10], [30, 250]]))
    list_of_obstacles.append(np.array([[50, 50], [80, 400]]))
    list_of_obstacles.append(np.array([[250, 250], [420, 260]]))
    list_of_obstacles.append(np.array([[250, 250], [260, 335]]))
    list_of_obstacles.append(np.array([[270, 210], [420, 220]]))
    list_of_obstacles.append(np.array([[390, 210], [420, 260]]))

    # Initial conditions of the vehicle.
    x_init = [325, 235]  # Initial position of the vehicle.
    v_init = [0, 0]  # Initial velocity of the vehicle.

    # Final conditions of the vehicle at the goal.
    x_goal = (390, 340)  # Goal position

    # Limit on parameters
    map_bound = np.array([[0, 0], [400, 350]])  # Map bounds (dx2 array with [lower_left, upper_right])
    v_max = 5  # Maximum velocity (scalar)
    u_max = 0.0  # Maximum input (scalar)

    r_plan = 150
    number_of_steps = 25
    NORMALS = 32
    DIMENSION = 2
    skip_factor = 1  # this is a little broken, keep it at 1 for now

    MAP = Scene(map_bound, list_of_obstacles, x_goal)
    MAP.show_scene()

    CONFIG = Config(16, 2, 25, 1e6)

    vehicle = Vehicle(x_init, v_init, v_max, u_max)

    model = make_model(MAP, CONFIG, vehicle)