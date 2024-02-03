""" Gazing code for UR5 robot. 
It minimizes the angle between two vectors, namely current gaze vector and desired gaze vector.

@project TUBITAK Kalfa  
@author Burak Bolat
@copyright '2022, Burak Bolat'
"""

# Robot related libraries
import rospy
import moveit_commander

import tf2_ros
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String
from visualization_msgs.msg import InteractiveMarkerFeedback
import utils

# Gaze related libraries
import numpy as np
from scipy.optimize import least_squares

import time

last_callback_time = 0

H_w1_w2 = np.array([[1, 0, 0, 0],
                    [0, 0, -1, -0.09465],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]])

H_w2_w3 = np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0.0823],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]])

def get_rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                    [np.sin(theta), np.cos(theta), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

def dist(x, gaze_point, curr_pose, r):
    h_start = H_w1_w2 @ get_rotation(x[1])
    h_end = h_start @ H_w2_w3  # Note that wrong rotation. Rotation of the last joint (wrist_3) omitted.
    
    h_start = r @ get_rotation(x[0]-curr_pose) @ h_start
    h_end = r @ get_rotation(x[0]-curr_pose) @ h_end

    start = h_start[:3, -1]
    end = h_end[:3, -1]
    current_dir = end - start
    current_dir /= np.linalg.norm(current_dir)

    gaze_point = gaze_point.T[:3].ravel()
    desired_dir = gaze_point - start
    desired_dir /= np.linalg.norm(desired_dir)

    dot_prod = np.inner(desired_dir, current_dir)
    cos_theta = dot_prod
    theta = np.arccos(cos_theta)
    return theta

def parallel_dist(x, curr_pose, prev_pose, r, show=False):
    h_start = H_w1_w2 @ get_rotation(curr_pose[4])
    h_end = h_start @ H_w2_w3 @ get_rotation(x[0])
    
    h_start = r @ get_rotation(curr_pose[3]-prev_pose) @ h_start
    h_end = r @ get_rotation(curr_pose[3]-prev_pose) @ h_end

    dot_prod = np.inner(np.array([0,0,1]), h_end[:3, 1])
    
    if show:
        print(h_end)
        print("--------++++++++++-w---------")
    
    return dot_prod

def gui_subs(data, args):
    if data.data == "Gaze": gaze(args)

def gaze(target_position):
    tick = time.time()

    transformation = tf_buffer.lookup_transform("base_link", "wrist_1_link", rospy.Time())
    r = R.from_quat(np.array([transformation.transform.rotation.x,
                            transformation.transform.rotation.y,
                            transformation.transform.rotation.z,
                            transformation.transform.rotation.w]))
    r = r.as_matrix()
    r = np.vstack((r, [0,0,0]))
    r = np.hstack((r, np.array([[transformation.transform.translation.x,
                            transformation.transform.translation.y,
                            transformation.transform.translation.z,
                            1.0]]).T))
    # gaze_point_pose = tf_buffer.lookup_transform("world", "human_wrist/base_link", rospy.Time())
    # gaze_point = gaze_point_pose.transform.translation 
    # gaze_point = np.array([gaze_point.x, gaze_point.y, gaze_point.z, 1])

    gaze_point = np.array(target_position + [1])


    # gaze_point = np.hstack((drawer.end, np.array([1])))
    curr_pose = group.get_current_joint_values()
    
    x0 = np.array([-1.57, -1.57])  # Elbow Down
    x0 = np.array([-4.0, -1.57])  # Elbow Up 1
    res = least_squares(dist, x0, method='dogbox', bounds=(-1.9*np.pi, 1.9*np.pi), args=(np.array([gaze_point]), curr_pose[3], r))
    theta_1, theta_2 = res['x']
    prev_pose = curr_pose[3]
    curr_pose[3] = theta_1
    curr_pose[4] = theta_2
    x0 = -1.57
    res = least_squares(parallel_dist, x0, method='dogbox', bounds=(-np.pi, np.pi), args=(curr_pose, prev_pose, r))
    curr_pose[5] = res['x'][0]
    # print(res)
    group.go(curr_pose)
    # parallel_dist([curr_pose[5]], curr_pose, prev_pose, r, True)

def gaze_vel(gaze_point, curr_pose, r):
    
    # x0 = np.array([-1.57, -1.57])  # Elbow Down
    x0 = np.array([-4.0, -1.57])  # Elbow Up 1
    # x0 = np.array([-1.27, -4.65])  # Elbow Up 2
    res = least_squares(dist, x0, method='dogbox', bounds=(-1.9*np.pi, 1.9*np.pi), args=(np.array([gaze_point]), curr_pose[3], r))
    theta_1, theta_2 = res['x']
    prev_pose = curr_pose[3]
    curr_pose[3] = theta_1
    curr_pose[4] = theta_2
    x0 = -1.57  # Elbow Down and Up 1
    # x0 = 1.57  # Elbow Up 2
    res = least_squares(parallel_dist, x0, method='dogbox', bounds=(-np.pi, np.pi), args=(curr_pose, prev_pose, r))
    # print(res)
    curr_pose[5] = res['x'][0]
    return curr_pose
    

def callback(data):
    global last_callback_time
    current_time = time.time()
    if current_time - last_callback_time > 0.5:  # 500 ms
        last_callback_time = current_time
        position = data.pose.position
        x = 0.387
        y = -0.386
        z = -0.305
        gaze([position.x, position.y, position.z])
    


if __name__=="__main__":
    rospy.init_node('gazing')
    group = moveit_commander.MoveGroupCommander("manipulator")
    home_config = [1.57, 0.78, -1.80, -1.57, -1.57, -1.57]  # Predefined home position
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    # group.go(home_config)
    # start = utils.pose2np(group.get_current_pose().pose)[:3]
    # drawer = utils.Drawer(start, start+np.array([0, 0.1, 0]), np.zeros((3)), np.zeros((3)))
    # drawer.create_sphere(1,(0,0,0),0.1,"/world", (0,255,255,0))
    print("I DREW!")
    # rospy.Subscriber("button_topic", String, gui_subs, (drawer))
    # rospy.Subscriber("button_topic", String, gui_subs, (drawer))

    input("Continue by pressing enter")
    # sub = rospy.Subscriber('/simple_marker/feedback', InteractiveMarkerFeedback, callback, queue_size=1)
    x = 0.387
    y = -0.386
    z = -0.305
    gaze([x, y, z])
    rate = rospy.Rate(120)  # Set the rate to 60 Hz
    exit()
    while not rospy.is_shutdown():
        rate.sleep()