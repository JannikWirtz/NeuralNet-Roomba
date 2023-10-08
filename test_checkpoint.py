import numpy as np
from NeuralNet import Net
from model import Model
from view import draw_simulation
import environments

colors = {1: 'r', 2: 'g', 3: 'b'}


for gen in [40]:
    folder_name = f'checkpoints/GA_bkp_mut=0.2_run2'
    genome = np.load(f'{folder_name}/best_candidate_gen={gen}.npy')

    # run simulation without plottings
    weights = [np.reshape(genome[:56] ,(4,14))]# first 56
    weights.append(np.reshape(genome[-8:] ,(2, 4))) # last 8
    nn = Net(weights)
    env = environments.getMediumLevel()
    env = environments.getEasyLevel()
    env = environments.getHardLevel()
    env = environments.getNarrowPathLevel()
    env = environments.getRoomLevel()


    # robot
    x_start = 5
    y_start = 5
    theta_start = np.pi/4
    robot_base_radius = 4
    sensors = 10
    robot = Model(x_start, y_start, theta_start, robot_base_radius, sensors)

    draw_simulation(robot, env, nn, draw=True, max_steps=500, alpha=genome[-2], time=genome[-1])