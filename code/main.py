import numpy as np
import sys
import pdb

from MapReader import MapReader
from MotionModel import MotionModel
from SensorModel import SensorModel
from SensorModel import SensorModel
from Resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time
import random

import yaml

""" Fix the randomness """
t = 11203
random.seed(t)
np.random.seed(t)

def visualize_map(occupancy_map, particles):
    fig = plt.figure()
    mng = plt.get_current_fig_manager();  # mng.resize(*mng.window.maxsize())
    plt.ion(); plt.imshow(occupancy_map, cmap='Greys'); plt.axis([0, 800, 0, 800]);
    # plt.scatter(particles[:,0]/10.0, particles[:,1]/10.0, c='b', marker='+')
    # plt.pause(0)


def visualize_timestep(X_bar, frame, tstep, frame_vis):
    x_locs = X_bar[:,0]/10.0
    y_locs = X_bar[:,1]/10.0
    scat = plt.scatter(x_locs, y_locs, c='b', marker='+')

    if frame_vis == 1:        
        fileStr = 'figs/frame_' + str(int(frame)) + '.png'
        plt.savefig(fileStr)

    plt.pause(tstep)
    scat.remove()

def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    # (randomly across the map) 
    y0_vals = np.random.uniform( 0, 7000, (num_particles, 1) )
    x0_vals = np.random.uniform( 3000, 7000, (num_particles, 1) )
    theta0_vals = np.random.uniform( -np.pi, np.pi, (num_particles, 1) )

    # initialize weights for all particles
    w0_vals = np.ones( (num_particles,1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals,y0_vals,theta0_vals,w0_vals))
    
    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles

    x_bounds, y_bounds = np.where(occupancy_map == 0)
    idx = np.random.choice(np.arange(len(x_bounds)), num_particles, replace=False)

    x0_vals = y_bounds[idx].reshape(len(idx), 1) * 10
    y0_vals = x_bounds[idx].reshape(len(idx), 1) * 10

    theta0_vals = np.random.uniform(-np.pi, np.pi, (num_particles, 1))

	# initialize weights for all particles    
    w0_vals = np.ones( (num_particles,1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def main():

    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]   
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """

    """
    Initialize Parameters
    """
    with open('config/params.yaml') as f:        
        params = yaml.load(f, Loader=yaml.FullLoader)

    src_path_map = params['map_path']
    src_path_log = params['data_path']

    """ Load the occupancy map """
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()

    """ Load the data log """
    logfile = open(src_path_log, 'r')

    """ Initialize the sub-modules """
    motion_model = MotionModel(params)
    sensor_model = SensorModel(occupancy_map, params)
    resampler = Resampling()

    num_particles = params['num_particles']
    adaptive_resampling_flag = params['adaptive_resampling_flag']
    X_bar = init_particles_freespace(num_particles, occupancy_map)

    vis_flag = params['map_vis']

    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if vis_flag:
        visualize_map(occupancy_map, X_bar[:,0:2])

    first_time_idx = True
    first_std = None

    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        meas_type = line[0]                                              # L : laser scan measurement, O : odometry measurement
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')   # convert measurement values from string to double

        odometry_robot = meas_vals[0:3]                                  # odometry reading [x, y, theta] in odometry frame
        time_stamp = meas_vals[-1]


        if (meas_type == "L"):
             odometry_laser = meas_vals[3:6]                             # [x, y, theta] coordinates of laser in odometry frame
             ranges = meas_vals[6:-1]                                    # 180 range measurement values from single laser scan
        
        print("Processing time step " + str(time_idx) + " at time " + str(time_stamp) + "s")

        if (first_time_idx):
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros( (num_particles,4), dtype=np.float64)
        u_t1 = odometry_robot

        for m in range(0, num_particles):

            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)


            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)                
                X_bar_new[m,:] = np.hstack((x_t1, w_t))
            else:
                X_bar_new[m,:] = np.hstack((x_t1, X_bar[m,3]))
        
                 
        X_bar = X_bar_new
        u_t0 = u_t1


        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)

        if adaptive_resampling_flag:
	        new_std_x, new_std_y = np.std(X_bar[:,0]), np.std(X_bar[:, 1])
	        new_std = np.sqrt(new_std_x**2 + new_std_y**2)
	        """
	        ADAPT PARTICLE SIZE
	        """
	        if (first_std == None):
	            first_std = new_std

	        new_num_particles = int(num_particles * (first_std/new_std))
	        
	        if new_num_particles <= np.min([1000, num_particles]):
	            num_particles = new_num_particles

	        # print('num_particles', num_particles)


        frame_vis = 0
        frame = 0        
        if vis_flag:
            frame = time_idx/10
            visualize_timestep(X_bar, frame, 0.00001, frame_vis)
            

if __name__=="__main__":
    main()