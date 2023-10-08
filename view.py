# Purely for visualization of the model
import matplotlib.pyplot as plt
import numpy as np
import math

def transformation_matrix(x, y, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])

def preprocess(sensors, alpha, time):
    base = 10
    return [base+(base*alpha-base)*(1-math.exp(-i*time)) for i in sensors]

def plot_dust(dust):
    for d in dust:
        plt.gcf().gca().add_artist(plt.Circle(
            d, 0.2, color='red', fill=True))

def plot_robot(robot):
    # Points expressed in the robot frame
    base_p = np.array([0, 0, 1]).T
    p1_i = np.array([0, 0, 1]).T
    p2_i = np.array([robot.r, 0, 1]).T

    # Points expressed in the world frame
    T = transformation_matrix(robot.x, robot.y, robot.theta)
    p1 = np.matmul(T, p1_i)
    p2 = np.matmul(T, p2_i)
    p_base = np.matmul(T, base_p)

    # Plot the robot base and its orientation
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
    plt.gcf().gca().add_artist(plt.Circle(
        p_base[:2], robot.r, color='k', fill=False))

    plt.xlim(0, 100)
    plt.ylim(0, 100)

def plot_sensor_values(robot):
    for i in range(len(robot.sensors)):
        angle = robot.sensors[i].angle
        text_distance_offset = 7
        x = text_distance_offset*np.cos(angle)
        y = text_distance_offset*np.sin(angle)
        s_p_i = np.array([x, y, 1]).T
        T = transformation_matrix(robot.x, robot.y, robot.theta)
        s_p1 = np.matmul(T, s_p_i)
        plt.text(s_p1[0], s_p1[1], round(robot.sensor_values[i]), fontsize='x-small',
                 horizontalalignment='center', verticalalignment='center')

def plot_robot_motor_speeds(robot):
    vl_p_i = np.array([0, 1.5, 1]).T
    vr_p_i = np.array([0, -1.5, 1]).T
    T = transformation_matrix(robot.x, robot.y, robot.theta)
    vl_p = np.matmul(T, vl_p_i)
    vr_p = np.matmul(T, vr_p_i)
    plt.text(vl_p[0], vl_p[1], round(robot.v_l), fontsize='x-small',
                horizontalalignment='center', verticalalignment='center')
    plt.text(vr_p[0], vr_p[1], round(robot.v_r), fontsize='x-small',
                horizontalalignment='center', verticalalignment='center')

def plot_environment(environment):
    for line in environment:
        plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'k')

def plot_robot_path(robot):
    for point in robot.path_history:
        plt.plot([point[0]], [point[1]], 'go', markersize=3)

def plot_stats(robot, total_dust_count, current_dust_count, max_steps, x=0, y=105):
    max_steps = 100 if (max_steps == -1) else max_steps
    plt.text(x, y, f'# of collisions: {robot.collision_counter}', fontsize='medium')
    plt.text(x, y+5, f'# of collected dust: {total_dust_count-current_dust_count}', fontsize='medium')

    plt.text(x+50, y, f'% of collisions: {robot.collision_counter/max_steps}', fontsize='medium')
    plt.text(x+50, y+5, f'% of collected dust: {round((total_dust_count-current_dust_count)/total_dust_count, 2)}', fontsize='medium')

def draw_simulation(robot, environment, nn=None, draw=True, max_steps=-1, alpha=0.2, time=0.5):
    fitness = 0

    if nn is None:
        # controls (only needs to run once) # Jannik
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: [
            exit(0) if event.key == 'escape' else None])
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: [
            robot.accelLeft() if event.key == 'e' else None])
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: [
            robot.brakeLeft() if event.key == 'd' else None])
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: [
            robot.accelRight() if event.key == 'i' else None])
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: [
            robot.brakeRight() if event.key == 'k' else None])
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: [
            robot.accelBoth() if event.key == 't' else None])
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: [
            robot.brakeBoth() if event.key == 'g' else None])
        plt.gcf().canvas.mpl_connect('key_press_event', lambda event: [
            robot.resetVelocities() if event.key == 'x' else None])
    
    # spread dust particels
    grid_step = 3
    dust_coords = [(x+grid_step,y+grid_step) for x in range(0,95, grid_step) for y in range(0,95, grid_step)]
    dust_count = len(dust_coords)

    step = 0
    while step < max_steps or max_steps == -1:
        step+=1

        robot.applyPhysicsStep(environment)

        if nn != None:
            output = nn.feedForward(preprocess(robot.sensor_values, alpha, time))
            
            # post process 
            robot.v_l = output[0]
            robot.v_r = output[1]

        
        # dust factor; reward for getting as much dust as quickly as possible
        for dust in dust_coords:
            x,y = dust
            if robot.x - robot.r < x and robot.x + robot.r > x and \
               robot.y - robot.r < y and robot.y + robot.r > y:
               dust_coords.remove(dust)
                
        # modify fitness (Evolutionary Robots paper)
        if robot.v_l !=0 or robot.v_r !=0 and not robot.collided_last:
            fitness += (robot.v_l+robot.v_r)/2*(1-np.sqrt(np.abs((np.abs(robot.v_l)-np.abs(robot.v_r)))/(np.abs(robot.v_l)+np.abs(robot.v_r))))*(1-np.max(robot.sensor_values)/200)


        if draw:
            plt.cla()# clear
            plot_robot(robot)
            plot_environment(environment)
            # plot_sensor_values(robot)
            # plot_robot_motor_speeds(robot)
            plot_dust(dust_coords)
            plot_robot_path(robot)
            plot_stats(robot, dust_count, len(dust_coords), max_steps)
            plt.pause(0.00000000001)

    # Compute collision and dust fitness
    dust_factor = 1-len(dust_coords)/dust_count
    fitness = fitness * math.pow((1-robot.collision_counter/max_steps),5) * math.pow((dust_factor),3)

    # fitness equalized for time ran (allows for comparison despite changing max t)    
    print("Fitness: " + str(fitness) + "\t collision%: " +\
         str(int(robot.collision_counter/max_steps*100)) + "\t dust%: " +str(int(dust_factor*100)))
    return fitness

if __name__ == '__main__':
    import train
    train()
