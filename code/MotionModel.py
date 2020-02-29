import sys
import numpy as np
import math as m 

class MotionModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self, params):

        """
        """
        self.alpha_1 = params['alpha_1']
        self.alpha_2 = params['alpha_2']
        self.alpha_3 = params['alpha_3']
        self.alpha_4 = params['alpha_4']


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]   
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        # u_t0    -> x_{t-1}^
        # u_t1    -> x_{t}^
        # x_{t-1} -> x_t0

        """
        """
        # [doc 1]: Sebastian Thurn, Probablistic Robotics - p 136
        # [doc 2]: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html
        # --------------------------------------------------------
        delta_rot_1 = m.atan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        delta_trans = m.sqrt((u_t1[0] - u_t0[0])**2 + (u_t1[1] - u_t0[1])**2)
        delta_rot_2 = u_t1[2] - u_t0[2] - delta_rot_1

        sigma_rot_1 = self.alpha_1*(delta_rot_1**2) + self.alpha_2*(delta_trans**2)
        sigma_trans = (self.alpha_3*(delta_trans**2)) + (self.alpha_4*(delta_rot_1**2)) + (self.alpha_4*(delta_rot_2**2))
        sigma_rot_2 = self.alpha_1*(delta_rot_2**2) + self.alpha_2*(delta_trans**2)

        cap_delta_rot_1 = delta_rot_1 - np.random.normal(0.0, sigma_rot_1)
        cap_delta_trans = delta_trans - np.random.normal(0.0, sigma_trans) 
        cap_delta_rot_2 = delta_rot_2 - np.random.normal(0.0, sigma_rot_2)
       
        x_t1 = np.zeros(3)
        x_t1[0] = x_t0[0] + cap_delta_trans*m.cos(x_t0[2] + cap_delta_rot_1)
        x_t1[1] = x_t0[1] + cap_delta_trans*m.sin(x_t0[2] + cap_delta_rot_1)
        x_t1[2] = x_t0[2] + cap_delta_rot_1 + cap_delta_rot_2

        return x_t1

if __name__=="__main__":
    pass
    
