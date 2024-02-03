#!/usr/bin/env python

""" This code plots the spline and interpolations.
It is written to warm up with splines.
Taken from https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html#spline-interpolation-in-1-d-object-oriented-univariatespline
"""

# Libraries for spline interpolation
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Libraries for robot motion
import rospy
import moveit_commander
import moveit_msgs.msg
import copy

def init(group_name):
    rospy.init_node('move_group_python_interface_tutorial',
                    anonymous=True)

    group = moveit_commander.MoveGroupCommander(group_name)
    return group

def show_spline_matplotlib(*pairs):
    plt.figure()
    plt.plot(pairs[0], pairs[1], 'x', pairs[2], pairs[3], pairs[2], np.sin(pairs[2]))
    plt.axis([-0.05, 3.33, -1.5, 1.5])
    plt.show()

if __name__=="__main__":
    
    group = init("manipulator")  # Initialize the arm group

    home_pose_joints = [1.57, 0.78, -1.57, -1.57, -1.57, 0]
    group.go(home_pose_joints)
    group.stop()

    # Interpolate Splines
    x = np.array([0, np.pi/3, 2*np.pi/3, np.pi])
    y = np.linspace(0, np.sin(np.pi), 4)
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(0, np.pi, np.pi/50)
    ynew = interpolate.splev(xnew, tck, der=0)
    show_spline_matplotlib(x, y, xnew, ynew)

