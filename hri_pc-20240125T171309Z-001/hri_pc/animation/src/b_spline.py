""" Implementation of B-Spline. 
Used for interpolating waypoints for arcing robot motions.
"""

import utils
import numpy as np
import copy

from scipy import interpolate

import math

def b_spline(start, end, num_sampling):
    control_points = sample_control_points(start, end, 2)
      
    tck, _ = interpolate.splprep(control_points, k=len(control_points[0,:])-1)
    ts = np.linspace(0, 1, num_sampling)
    x_spl, y_spl, z_spl = interpolate.splev(ts, tck)
    
    return np.stack((x_spl, y_spl, z_spl))

def sample_control_points(start, end, arc_type=0):
    dist = np.linalg.norm(end-start)
    start_pose = utils.np_pose2ros_pose(start)

    vector_y = utils.get_gripper_vector(start_pose, np.array([0, 1, 0, 1]))
    vector_z = utils.get_gripper_vector(start_pose, np.array([0, 0, 1, 1]))

    if arc_type == 0:      
        nodes =  np.zeros((3,4))
        nodes[:,0] = [start[0], start[1], start[2]]
        nodes[:,3] = [end[0], end[1], end[2]]
        nodes[:,1] = start[:3] + vector_y*(dist/math.sqrt(2))
        nodes[:,2] = start[:3] + vector_y*(dist/math.sqrt(2)) + vector_z*dist

    elif arc_type == 1:
        nodes =  np.zeros((3,4))
        nodes[:,0] = [start[0], start[1], start[2]]
        nodes[:,3] = [end[0],end[1],end[2]]
        nodes[:,1] = start[:3] + vector_y*(dist/(math.sqrt(2)*2))
        nodes[:,2] = start[:3] + vector_y*(dist/(math.sqrt(2)*2)) + vector_z*(dist/2)

    # Do not use if you do not know what you are doing
    elif arc_type == 2:
        nodes =  np.zeros((3,5))
        nodes[:,0] = [start[0], start[1], start[2]]
        nodes[:,4] = [end[0],end[1],end[2]]
        nodes[:,1] = start[:3] + vector_y*(dist/(math.sqrt(2)*2))
        nodes[:,2] = start[:3] + vector_y*(dist/(math.sqrt(2)*2)) + vector_z*(dist/2)
        nodes[:,3] = start[:3] + vector_z*(dist*(3/2))

    return nodes