""" This code implements the robot gaze with positional control interface.
@author Burak Bolat
"""

# Robot related libraries
import rospy
import moveit_commander

import tf2_ros
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose

import numpy as np

import utils

import numpy as np


if __name__=="__main__":
    rospy.init_node('gazing')
    group = moveit_commander.MoveGroupCommander("manipulator")
    print(len(group.get_jacobian_matrix(group.get_current_joint_values())))
    home_config = [1.57, 0.78, -1.80, -1.57, -1.57, -1.57]  # Predefined home position
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    group.go(home_config)
    start = utils.pose2np(group.get_current_pose().pose)[:3]
    drawer = utils.Drawer(start, start+np.array([0, 0, 0.2]), np.zeros((3)), np.zeros((3)))

    transformation = tf_buffer.lookup_transform("world", "tool0", rospy.Time())
    print(transformation)
    rotation = transformation.transform.rotation
    r = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
    gaze_curr = np.array([r.as_matrix()[0][2], r.as_matrix()[1][2], r.as_matrix()[2][2]])
    print("Gaze curr:", gaze_curr, np.linalg.norm(gaze_curr))
    gaze_goal = drawer.end - start
    gaze_goal /= np.linalg.norm(gaze_goal)
    print("Gaze goal:", gaze_goal)
    
    v = np.cross(gaze_goal, gaze_curr)
    s = np.linalg.norm(v)
    c = np.dot(gaze_goal, gaze_curr)
    print("s, v, c", s, v, c)
    v_crossed = np.matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot_mat = np.identity(3) + v_crossed + np.dot(v_crossed, v_crossed)/(1+c)
    # rot_mat = R.from_matrix(rot_mat.T)
    r = np.matmul(rot_mat.T, r.as_matrix())
    print("Rotation", r)

    

    r = R.as_quat(R.from_matrix(r))

    pose_goal = Pose()
    pose_goal.position.x = start[0]
    pose_goal.position.y = start[1]
    pose_goal.position.z = start[2]
    pose_goal.orientation.x = r[0]
    pose_goal.orientation.y = r[1]
    pose_goal.orientation.z = r[2]
    pose_goal.orientation.w = r[3]

    group.set_pose_target(pose_goal)
    group.go()
    
    while not rospy.is_shutdown():
        rospy.spin()