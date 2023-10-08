import numpy as np


class Sensor:
    MAX_RANGE = 200

    def __init__(self, angle):
        # relative to robots front
        self.angle = angle

    def getDistance(self, robot, environment):
        # robots angle + angle relative to robot front
        # cast ray from (x, y) with angle, for max_range
        min_distance = self.MAX_RANGE
        for line in environment:
            tmp = lineRayIntersectionPoint(robot, self.angle, line[0], line[1])
            min_distance = np.min([min_distance, tmp])
        return min_distance - robot.r


def norm(vector):
    magnitude = np.sqrt(np.dot(np.array(vector), np.array(vector)))
    return np.array(vector)/magnitude


def lineRayIntersectionPoint(robot, sensor_angle, point1, point2):
    # Convert to numpy arrays
    origin = np.array((robot.x, robot.y), dtype=np.float)
    direction = [np.cos(robot.theta + sensor_angle),
                 np.sin(robot.theta + sensor_angle)]
    direction = np.array(norm(direction), dtype=np.float)
    point1 = np.array(point1, dtype=np.float)
    point2 = np.array(point2, dtype=np.float)

    # check for intersection of ray and line
    v1 = origin - point1
    v2 = point2 - point1
    v3 = np.array([-direction[1], direction[0]])
    t1 = np.cross(v2, v1) / np.dot(v2, v3)
    t2 = np.dot(v1, v3) / np.dot(v2, v3)
    if t1 >= 0.0 and t2 >= 0.0 and t2 <= 1.0:
        # return euclidean distance
        return np.sqrt(np.dot(np.array(t1 * direction), np.array(t1 * direction)))
    return Sensor.MAX_RANGE
