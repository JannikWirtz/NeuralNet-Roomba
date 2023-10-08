# Modeling of the Robot (data like positions and modifications to it)
import numpy as np
from shapely.geometry import LineString, Point

from sensor import Sensor


class Model:

    ACCELERATION = 20

    def __init__(self, x_start, y_start, theta_start, robot_base_radius, num_sensors):
        self.collision_counter = 0
        self.collided_last = False
        self.x = x_start
        self.y = y_start
        self.theta = theta_start
        self.path_history = [(x_start, y_start)]
        self.r = robot_base_radius
        self.l = self.r * 2.0
        self.v_l = 0
        self.v_r = 0
        self.sensors = [Sensor(i*2*np.pi/num_sensors)
                        for i in range(num_sensors)]
        self.sensor_values = [0.0 for i in range(num_sensors)]

    def getSensorData(self, environment):
        return [sensor.getDistance(self, environment) for sensor in self.sensors]

    def accelLeft(self): 
        self.v_l += self.ACCELERATION

    def accelRight(self):
        self.v_r += self.ACCELERATION

    def accelBoth(self):   
        self.v_l += self.ACCELERATION
        self.v_r += self.ACCELERATION

    def brakeLeft(self):   
        self.v_l -= self.ACCELERATION

    def brakeRight(self):   
        self.v_r -= self.ACCELERATION

    def brakeBoth(self):   
        self.v_l -= self.ACCELERATION
        self.v_r -= self.ACCELERATION

    def isColliding(self, environment):   
        p = Point(self.x, self.y)
        c = p.buffer(self.r).boundary

        intersections = []
        lines = []
        for line in environment:
            i = c.intersection(LineString(line))

            if i.geom_type == 'Point':
                intersections.append(i.coords)
                lines.append(line)
            elif i.geom_type == 'MultiPoint':
                intersections.append(i.geoms[0].coords)
                intersections.append(i.geoms[1].coords)
                lines.append(line)

        intersections = [np.array((geom.xy[0][0], geom.xy[1][0]))
                         for geom in intersections]
        # print(intersections)
        return intersections, lines

    def applyPhysicsStep(self, environment):

        self.collided_last = False
        # update sensor data for current position
        self.sensor_values = self.getSensorData(environment)

        speeds = []
        limit = 300
        
        vl_tmp = self.v_l
        vr_tmp = self.v_r

        while np.abs(vl_tmp) > 0 or np.abs(vr_tmp) > 0:
            if np.abs(vl_tmp) > limit and np.abs(vr_tmp) > limit:

                if np.abs(vl_tmp) > np.abs(vr_tmp):
                    factor = np.abs(vr_tmp)/np.abs(vl_tmp)

                    for _ in range(int(np.abs(vl_tmp)/limit)):
                        speeds.append((limit* np.sign(vl_tmp), factor * limit * np.sign(vr_tmp)))
                        vl_tmp -= limit * np.sign(vl_tmp)
                        vr_tmp -= limit * np.sign(vr_tmp)*factor

                elif np.abs(vl_tmp) < np.abs(vr_tmp):
                    factor = np.abs(vl_tmp)/np.abs(vr_tmp)

                    for _ in range(int(np.abs(vr_tmp)/limit)):
                        speeds.append((limit* np.sign(vl_tmp), factor * limit * np.sign(vr_tmp)))
                        vl_tmp -= limit * np.sign(vl_tmp)*factor
                        vr_tmp -= limit * np.sign(vr_tmp)
                
                else:
                    for _ in range(int(np.abs(vr_tmp)/limit)):
                        speeds.append((limit* np.sign(vl_tmp), limit * np.sign(vr_tmp)))
                        vl_tmp -= limit * np.sign(vl_tmp)
                        vr_tmp -= limit * np.sign(vr_tmp)

            elif vl_tmp > limit:
                speeds.append((limit* np.sign(vl_tmp), vr_tmp))
                vl_tmp -= limit * np.sign(vl_tmp)
                vr_tmp = 0
            elif vr_tmp > limit:
                speeds.append((vl_tmp, limit * np.sign(vr_tmp)))
                vl_tmp = 0
                vr_tmp -= limit * np.sign(vr_tmp)
            else:
                speeds.append((vl_tmp, vr_tmp))
                vl_tmp = 0
                vr_tmp = 0

        #print(f'Velocity: {self.v_l}, {self.v_r}')
        for (v_l, v_r) in speeds:

            self.theta = self.theta % (2*np.pi)  # limit theta
            # update sensor data for current position
            self.sensor_values = self.getSensorData(environment)

            dt = 0.01

            #print(f'Angle: {self.theta}')

            dx = -1
            dy = -1

            # Change Position according to velocities  
            if np.abs(v_r-v_l) < 1e-8: # looking for exact match causes robot not to move in else case
                dx = v_r * np.cos(self.theta) * dt
                dy = v_r * np.sin(self.theta) * dt
            else:
                R = (self.l/2) * (v_l + v_r)/(v_r - v_l)
                omega = (v_r - v_l)/self.l

                ICC = np.array([(self.x - R * np.sin(self.theta)),
                            (self.y + R * np.cos(self.theta))])
                rotMat = np.array([
                    [np.cos(omega*dt), -np.sin(omega*dt), 0],
                    [np.sin(omega*dt), np.cos(omega*dt), 0],
                    [0, 0, 1]
                ])
                posMat = np.array([
                    [self.x - ICC[0]],
                    [self.y - ICC[1]],
                    [self.theta]
                ])
                transMat = np.array([
                    [ICC[0]],
                    [ICC[1]],
                    [omega*dt]
                ])

                pose = np.matmul(rotMat, posMat) + transMat

                dx = pose[0][0] - self.x
                dy = pose[1][0] - self.y
                self.theta = pose[2][0]

            prev_x = self.x
            prev_y = self.y
            # apply changes, look for collision afterwards
            self.x += dx
            self.y += dy

            intersections, lines = self.isColliding(environment)

            # merge collision coordinates
            x_collisions = []
            y_collisions = []

            for x, y in intersections:
                if x % 1 == 0.0:
                    x_collisions.append(x)

                if y % 1 == 0.0:
                    y_collisions.append(y)

            # remove duplicates from multipoints
            x_collisions = list(set(x_collisions))
            y_collisions = list(set(y_collisions))

            anti_jump = 0.01
            # corner
            if len(x_collisions) > 0 and len(y_collisions) > 0:
                self.collision_counter +=1
                self.collided_last = True
                # take back partial timestep and recalc positon
                x_offset = self.x - x_collisions[0]
                y_offset = self.y - y_collisions[0]

                offset_sign_x = np.sign(x_offset)
                offset_sign_y = np.sign(y_offset)

                left_blocked = False
                right_blocked = False
                below_blocked = False
                above_blocked = False

                for line in lines:   
                    line = np.array(line)
                    # horizontal or vertical
                    if np.abs(line[1] - line[0])[1] == 0:  # horizontal line
                        # at relevant width
                        if self.x - self.r/2 < np.max([line[0][0], line[1][0]]) and self.x + self.r/2 > np.min([line[0][0], line[1][0]]):
                            if line[0][1] > self.y - self.r/2:  # above robot
                                above_blocked = True
                            elif line[0][1] < self.y + self.r/2:  # below robot
                                below_blocked = True
                    elif np.abs(line[1] - line[0])[0] == 0:  # vertical line
                        # at relevant height
                        if self.y - self.r/2 < np.max([line[0][1], line[1][1]]) and self.y + self.r/2 > np.min([line[0][1], line[1][1]]):
                            if line[0][0] > self.x - self.r/2:  # right to robot
                                right_blocked = True
                            elif line[0][0] < self.x + self.r/2:  # left to robot
                                left_blocked = True
                    # else:# diagonal

                # slide
                if right_blocked and above_blocked or left_blocked and above_blocked or left_blocked and below_blocked or right_blocked and below_blocked:
                    self.x = x_collisions[0] + offset_sign_x*(self.r+anti_jump)
                    self.y = y_collisions[0] + offset_sign_y*(self.r+anti_jump)
                elif (right_blocked or left_blocked) and not above_blocked and not below_blocked:
                    self.x = x_collisions[0] + offset_sign_x*(self.r+anti_jump)
                elif (above_blocked or below_blocked) and not left_blocked and not right_blocked:
                    self.y = y_collisions[0] + offset_sign_y*(self.r+anti_jump)
                else:  # exact corner match?
                    self.x = prev_x + offset_sign_x*(anti_jump) + dt*dx
                    self.y = prev_y + offset_sign_y*(anti_jump) + dt*dy

            # horizontal  
            elif len(y_collisions) > 0:
                self.collision_counter +=1
                self.collided_last = True

                y_offset = self.y - y_collisions[0]
                offset_sign = np.sign(y_offset)
                self.y = y_collisions[0] + offset_sign * \
                    (self.r+anti_jump)  # reset y

                # add to x according to angle to wall (only considering horizontal walls)
                if self.theta >= 0 and self.theta <= np.pi/2:  # top-right quarter of circle
                    #print(((np.pi/2-self.theta)/(np.pi/2)))
                    self.x += dt*dy * ((np.pi/2-self.theta)/(np.pi/2))
                elif self.theta <= np.pi and self.theta > np.pi/2:  # top-left quarter of circle
                    #print((self.theta/(np.pi/2)-1))
                    self.x += dt*dy * (self.theta/(np.pi/2)-1)
                elif self.theta <= 3/2*np.pi and self.theta > np.pi:  # bottom-left quarter of circle
                    #print((-(self.theta-3/2*np.pi)/(np.pi/2)))
                    self.x += dt*dy * (-(self.theta-3/2*np.pi)/(np.pi/2))
                elif self.theta <= 2*np.pi and self.theta > 3/2*np.pi:  # bottom-right quarter of circle
                    #print(1-(2*np.pi-(self.theta))/(np.pi/2))
                    self.x += dt*dy * (1-(2*np.pi-(self.theta))/(np.pi/2))

            # vertical   
            elif len(x_collisions) > 0:
                self.collision_counter +=1
                self.collided_last = True
                
                x_offset = self.x - x_collisions[0]
                offset_sign = np.sign(x_offset)
                self.x = x_collisions[0] + offset_sign * \
                    (self.r+anti_jump)  # reset x

                # add to x according to angle to wall (only considering horizontal walls)
                if self.theta >= 0 and self.theta <= np.pi/2:  # top-right quarter of circle
                    self.y += dt*dx * np.abs(((np.pi/2-self.theta)/(np.pi/2)-1))
                elif self.theta <= np.pi and self.theta > np.pi/2:  # top-left quarter of circle
                    self.y += dt*dx * np.abs((self.theta/(np.pi/2)-1)-1)
                elif self.theta <= 3/2*np.pi and self.theta > np.pi:  # bottom-left quarter of circle
                    self.y += dt*dx * np.abs((-(self.theta-3/2*np.pi)/(np.pi/2))-1)
                elif self.theta <= 2*np.pi and self.theta > 3/2*np.pi:  # bottom-right quarter of circle
                    self.y += dt*dx * np.abs((1-(2*np.pi-(self.theta))/(np.pi/2))-1)

            self.path_history.append([self.x, self.y])

    def resetVelocities(self):   
        self.v_l = 0
        self.v_r = 0
