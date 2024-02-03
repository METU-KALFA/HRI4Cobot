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
import rosbag
import os

from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA




if __name__ == "__main__":
    
    rospy.init_node("Demo")
    # İnit MoveIt
    # group = moveit_commander.MoveGroupCommander("manipulator")
    # tf_buffer = tf2_ros.Buffer()
    # tf_listener = tf2_ros.TransformListener(tf_buffer, buff_size=120)
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=1)
    r = 20 # Rate
    rate = rospy.Rate(r)
    time.sleep(1)


    nod_1 = np.sin(np.linspace(-np.pi,0,20))
    nod_2 = np.sin(np.linspace(0,np.pi, 40))
    nod_3 = np.sin(np.linspace(0, -np.pi,20))
    nod = np.concatenate((nod_1, nod_2, nod_3))
    nod_length = nod.shape[0]


    


    flag = False
    i = 0
    while not rospy.is_shutdown():
        vel_msg = Float64MultiArray()
        
        if not flag: 
            input("Should I nod my head ? ")
            flag = True
        else :
            # Negative Nod

            # joint_velocities = np.array([0, 0, 0, 0, nod[i], 0])
            # vel_msg.data = joint_velocities
            # pub.publish(vel_msg)

            # i += 1
            # i = i % nod_length
            # if i == 0:
            #     vel_msg.data = np.array([0, 0, 0, 0, 0, 0])
            #     pub.publish(vel_msg)
            #     rate.sleep()
            #     exit()

            joint_velocities = np.array([0, 0, 0, nod[i] / 3, 0, 0])
            vel_msg.data = joint_velocities
            pub.publish(vel_msg)

            i += 1
            i = i % nod_length
            if i == 0:
                vel_msg.data = np.array([0, 0, 0, 0, 0, 0])
                pub.publish(vel_msg)
                rate.sleep()
                exit()
        rate.sleep()
            

        

