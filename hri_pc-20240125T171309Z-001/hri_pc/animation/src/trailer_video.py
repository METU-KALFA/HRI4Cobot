# Robot related libraries
import rospy
import moveit_commander
import numpy as np
import actionlib
import tf2_ros
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64MultiArray, String, Float64, Int16, Int16MultiArray
from controller_manager_msgs.srv import SwitchController
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

from simple_pid import PID
from gaze_head_positional import gaze_vel
import copy
import time
import yaml
import random 
import os

from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import random


def reset_pids(Kp=1.0):
    # Kp, Ki, Kd = (5.0, 0.1, 0.1)
    # Kp, Ki, Kd = (3.0, 0.05, 0.0)
    Ki, Kd = (0.00, 0.01)
    pid_w1 = PID(Kp, Ki, Kd, setpoint=0.0)
    pid_w2 = PID(Kp, Ki, Kd, setpoint=0.0)
    pid_w3 = PID(Kp, Ki, Kd, setpoint=0.0)
    return [pid_w1, pid_w2, pid_w3]


def gaze(group, tf_buffer, pids, target=None):
    joint_states = group.get_current_joint_values()
    try:
        transformation = tf_buffer.lookup_transform("world", "wrist_1_link", rospy.Time())
        rot = R.from_quat(np.array([transformation.transform.rotation.x,
                                transformation.transform.rotation.y,
                                transformation.transform.rotation.z,
                                transformation.transform.rotation.w]))
        rot = rot.as_matrix()
        rot = np.vstack((rot, [0,0,0]))
        rot = np.hstack((rot, np.array([[transformation.transform.translation.x,
                                transformation.transform.translation.y,
                                transformation.transform.translation.z,
                                1.0]]).T))
        if target is None:
            gaze_point_pose = tf_buffer.lookup_transform("world", "eye", rospy.Time())
            gaze_point = gaze_point_pose.transform.translation
            gaze_point = np.array([gaze_point.x, gaze_point.y, gaze_point.z, 1])
        else:
            target = np.array(target)
            gaze_point = np.hstack((target, [0]))
        desired_joints = np.array(gaze_vel(gaze_point, copy.deepcopy(joint_states), rot))
    except Exception as e:
        desired_joints = np.array([0, 0, 0, joint_states[3], joint_states[4], joint_states[5]])    
    
    pids[0].setpoint = desired_joints[3]
    pids[1].setpoint = desired_joints[4]
    pids[2].setpoint = desired_joints[5]
    gaze_vels = np.array([pids[0](joint_states[3]), pids[1](joint_states[4]), pids[2](joint_states[5])])
    return gaze_vels

def breathe(breathe_dict, i, gaze=True):
    shape_vel = breathe_dict["shape_vel"]
    num_of_vel_pts = breathe_dict["num_of_vel_pts"]
    indices = breathe_dict["indices"]
    amplitude = breathe_dict["amplitude"]
    bpm = breathe_dict["bpm"]
    breathe_vec = breathe_dict["breathe_vec"]
    r = breathe_dict["r"]

    # Interpolate magnitude of velocity
    vel_mag = np.interp(num_of_vel_pts*bpm*i/r, indices, shape_vel)
    vel = breathe_vec * vel_mag * bpm * amplitude  # x = vt --> ax = (av)t
    joint_states = group.get_current_joint_values()
    jacobian = group.get_jacobian_matrix(joint_states)  # np array 6x6

    # rcond may be tuned not to get closer to singularities.
    # Take psuedo inverse of Jacobian.
    pinv_jacobian = np.linalg.pinv(jacobian, rcond=1e-15)  
    if gaze: joint_vels = np.dot(pinv_jacobian[:3], vel)
    else: joint_vels = np.dot(pinv_jacobian, vel)
    return joint_vels


if __name__=="__main__":
    # Initialization
    rospy.init_node("experiment")
    group = moveit_commander.MoveGroupCommander("manipulator")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, buff_size=120)
    
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=1)
    while not pub.get_num_connections(): rospy.sleep(0.1)
    
    r = 30  # Rate (Frequency)
    rate = rospy.Rate(r)
    time.sleep(5)
    

    # Set breathing parameters
    breathe_dict = {}
    trans = tf_buffer.lookup_transform('base_link', 'world', rospy.Time(0))
    quat = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
    roti = R.from_quat(quat)
    # roti = np.dot(roti.as_matrix(), [-0.3,0.7,0])  # Breathe dir in ee frame
    roti = np.dot(roti.as_matrix(), [-0.6,0,0.4])  # Breathe dir in world frame
    breathe_vec = np.concatenate((roti, [0,0,0]))

    # Human data integrated, take difference
    shape_data = np.array([0.0, 0.005624999999999991, 0.014378906250000045, 
            0.024486547851562568, 0.03474537368774422, 0.0443933953943253, 
            0.0529963630275786, 0.060354717840215955, 0.06642930274659942, 
            0.07128390374926041, 0.07504226099316191, 0.0778570788273204, 
            0.07988866580429121, 0.08129106189085289, 0.08220379893995788, 
            0.08274774841852639, 0.08302380913885798, 0.0790451566321223, 
            0.07260624098380641, 0.06495160665441868, 0.05691987986583036, 
            0.04905405662967122, 0.041685214275588134, 0.039685214275588134, 0.03499542709050685, 
            0.02906453874530568, 0.02390450659228971, 0.019484261167040162, 
            0.015747394717907204, 0.012624483451303736, 0.010041439737111357, 
            0.007924965428885544, 0.006205920714800195, 0.004821221732756786, 
            0.0037147237823418333, 0.002837426374215357, 0.0021472441754255556, 
            0.0016085180924106934, 0.0011913883859058227, 0.000871112900045046, 
            0.0006273850763798272, 0.00044368593276999935])

    # shape_vel = np.array([shape_data[i+1]-shape_data[i] for i in range(len(shape_data)-1)])
    shape_vel = np.array([shape_data[i+1]-shape_data[i] for i in range(len(shape_data)-1)])
    shape_vel = np.hstack(([shape_data[-1]-shape_data[0]], shape_vel))
    # shape_vel = np.array(shape_vel)
    num_of_vel_pts = shape_vel.shape[0]
    indices = np.linspace(0, num_of_vel_pts-1, num_of_vel_pts)
    breathe_dict["shape_vel"] = shape_vel
    breathe_dict["num_of_vel_pts"] = num_of_vel_pts
    breathe_dict["indices"] = indices

    period = 4  # Period of a breathing
    # Breathing parameters: Amplitude, Frequency (bpm), Total Duration
    amplitude = 60.0
    bpm = 1.0/period
    i=1
    breathe_dict["amplitude"] = amplitude
    breathe_dict["bpm"] = bpm
    breathe_dict["breathe_vec"] = breathe_vec
    breathe_dict["r"] = r

    Kp = 8.0
    pids = None
    pids2 = None
    
    # Moving average setting for breathing
    moving_vels = []
    window_size = 20
    vel_cum = [0., 0., 0.]
    pids = reset_pids(Kp)
    tick = time.time()

    while not rospy.is_shutdown():
        if time.time() - tick > 10: 
            gaze_vels = gaze(group, tf_buffer, pids)
            q_max = 1.3
            if np.linalg.norm(gaze_vels) > q_max: gaze_vels *= q_max / np.linalg.norm(gaze_vels)
        
        else:
            gaze_vels = np.array([0, 0, 0])
        breathe_vels = breathe(breathe_dict, i, gaze=True)
        i += 1
        i = i%((r/bpm)+1)

        if len(moving_vels) == window_size:
            moving_vels.pop(0)
        moving_vels.append(breathe_vels)
        vel_cum = sum(moving_vels) / len(moving_vels)
        breathe_vels = vel_cum

        joint_vels = np.concatenate((breathe_vels, gaze_vels))
        # joint_vels = np.concatenate(([0, 0, 0], gaze_vels))

        vel_msg = Float64MultiArray()
        vel_msg.data = joint_vels
        pub.publish(vel_msg)
        rate.sleep()
