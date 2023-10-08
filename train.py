import numpy as np

from model import Model
import environments
from view import draw_simulation

x_start = 5
y_start = 5
theta_start = np.pi/4
robot_base_radius = 4
sensors = 10
robot = Model(x_start, y_start, theta_start, robot_base_radius, sensors)
#env = environments.getEasyLevel()
#env = environments.getMediumLevel()
env = environments.getHardLevel()
# env = environments.getNarrowPathLevel()
# env = environments.getRoomLevel()
# env = environments.getLabyrinth()
draw_simulation(robot, env)
