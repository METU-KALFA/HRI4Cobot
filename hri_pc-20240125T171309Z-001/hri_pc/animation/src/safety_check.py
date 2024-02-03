import rospy
import moveit_commander
import numpy as np
import actionlib
import tf2_ros
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64MultiArray, String, Float64
from controller_manager_msgs.srv import SwitchController
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

from simple_pid import PID
from gaze_head_positional import gaze_vel
import copy
import time
import yaml

def connect_service(name):
    client = actionlib.SimpleActionClient(name,
                                            FollowJointTrajectoryAction)
    print(client.wait_for_server(timeout=rospy.Duration(1.0)))
    print("Connected to trajectory manager")
    return client

def create_traj(waypoints):
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ["elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    
    for waypoint, tfs in waypoints:
        goal.trajectory.points.append(JointTrajectoryPoint())
        goal.trajectory.points[-1].positions = waypoint
        goal.trajectory.points[-1].time_from_start = rospy.Duration(float(tfs))

    goal.trajectory.header.stamp = rospy.Time.now()
    return goal

if __name__=="__main__":
    rospy.init_node("experiment")
    group = moveit_commander.MoveGroupCommander("manipulator")
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=1)
    client = connect_service("pos_joint_traj_controller/follow_joint_trajectory")
    switch_controller = rospy.ServiceProxy(
                       '/controller_manager/switch_controller', SwitchController)
    
    # switch_controller(['pos_joint_traj_controller'],['joint_group_vel_controller'], 2, False, 1)
    switch_controller(['joint_group_vel_controller'],['pos_joint_traj_controller'], 2, False, 1)

    waypoints = [[[1.699819564819336, -1.222046200429098, 2.042504072189331, -2.9689176718341272, -1.3403967062579554, -1.7585051695453089], 2],
    [[1.5558209419250488, -1.081384007130758, 2.0427796840667725, -2.9688456694232386, -1.3404086271869105, -1.758517090474264], 4],
    [[1.699819564819336, -1.222046200429098, 2.042504072189331, -2.9689176718341272, -1.3403967062579554, -1.7585051695453089], 6]]
    # [[1.4420933723449707, -1.3163026014911097, 0.7272219061851501, -1.752918545399801, -1.5200727621661585, -1.537652317677633], 10]]
    
    # trajectory = create_traj(waypoints)
    # print(trajectory)
    # client.send_goal(trajectory)
    # result = client.wait_for_result(rospy.Duration(20.0))
    
    joint_vel = Float64MultiArray()
    joint_vel.data = [0., 0.1, -0.1, 0., 0., 0.]
    pub.publish(joint_vel)

    rospy.spin()
    