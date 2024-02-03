import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import CollisionObject
import geometry_msgs.msg
import numpy as np
from math import *
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf
import argparse
import actionlib
import control_msgs.msg
from trajectory_msgs.msg import *
import matplotlib.pyplot as plt
import os
import subprocess
from moveit_msgs.msg import CollisionObject
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point
from scipy import interpolate
from mpl_toolkits import mplot3d


#initialize the robot and simulation enviroment
def initialize_moveit(group_name):
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_python_interface_tutorial',
                anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    group = moveit_commander.MoveGroupCommander(group_name)
   
    return group,robot,scene

#move robot to an inital state, used to orinet grip for now
def move_init_state(group):
    joint_goal = group.get_current_joint_values()
    joint_goal[5] = pi/2
    group.set_joint_value_target(joint_goal)
    plan = group.plan()
    group.execute(plan,wait=True)
    return plan 

def get_gripper_orient_vec(group,orient_vec):
    wpose = group.get_current_pose().pose
    matrixx = tf.transformations.quaternion_matrix([wpose.orientation.x,wpose.orientation.y,wpose.orientation.z,wpose.orientation.w])
    inv_matrixx = tf.transformations.inverse_matrix(matrixx)
    vectorx = np.matmul(orient_vec,inv_matrixx)
    vectorx = tf.transformations.unit_vector([vectorx[0],vectorx[1],vectorx[2]])
    return vectorx,wpose


def b_spline_arcing(group,target, num_sampl_points = 200 ,arcing_type = 1,Print = True):
    vectorz, wpose = get_gripper_orient_vec(group,[0,0,1,1])
    vectory, _ = get_gripper_orient_vec(group,[0,1,0,1])
    x_r, y_r, z_r = wpose.position.x, wpose.position.y, wpose.position.z
    robot_position = np.array([x_r,y_r,z_r])
    dist = np.linalg.norm(robot_position - target)

    if arcing_type == 1:      
        nodes =  np.zeros((3,4))
        nodes[:,0] = [x_r, y_r, z_r]
        nodes[:,3] = [target[0],target[1],target[2]]
        nodes[:,1] = robot_position + vectory*(dist/sqrt(2))
        nodes[:,2] = robot_position + vectory*(dist/sqrt(2)) + vectorz*dist
    
    elif arcing_type == 2:
        nodes =  np.zeros((3,4))
        nodes[:,0] = [x_r, y_r, z_r]
        nodes[:,3] = [target[0],target[1],target[2]]
        nodes[:,1] = robot_position + vectory*(dist/(sqrt(2)*2))
        nodes[:,2] = robot_position + vectory*(dist/(sqrt(2)*2)) + vectorz*(dist/2)

    elif arcing_type == 3:
        nodes =  np.zeros((3,5))
        nodes[:,0] = [x_r, y_r, z_r]
        nodes[:,4] = [target[0],target[1],target[2]]
        nodes[:,1] = robot_position + vectory*(dist/(sqrt(2)*2))
        nodes[:,2] = robot_position + vectory*(dist/(sqrt(2)*2)) + vectorz*(dist/2)
        nodes[:,3] = robot_position + vectorz*(dist*(3/2))

    k_param = len(nodes[0,:])-1
    print(nodes.shape)
    print([nodes[0,:],nodes[1,:],nodes[2,:]])
    tck, _ = interpolate.splprep([nodes[0,:],nodes[1,:],nodes[2,:]], k=k_param)
    sample_vector = np.linspace(0,1,num_sampl_points)
    x_spl, y_spl, z_spl = interpolate.splev(sample_vector, tck)
    waypoints = []
    wpose_next = copy.deepcopy(wpose)
    for i in range(len(x_spl)):
        wpose_next.position.x = x_spl[i]
        wpose_next.position.y = y_spl[i]
        wpose_next.position.z = z_spl[i]
        waypoints.append(copy.deepcopy(wpose_next))
    
    if Print:
        ax = plt.axes(projection='3d')
        ax.scatter3D(nodes[0,:], nodes[1,:], nodes[2,:], c=nodes[2,:], marker = "o")
        ax.plot(x_spl, y_spl, z_spl, 'g', label = 'B-Spline Curve')
        ax.plot(nodes[0,:], nodes[1,:], nodes[2,:], 'gray',label='Original Points')
        ax.legend()
        plt.show() 
    
    return waypoints

def draw_sphere(group,scene,robot,sphere_name):
    vectorx, _ = get_gripper_orient_vec(group,[0,0,1,1])
    pos_p = group.get_current_pose()
    pos_p.header.frame_id = robot.get_planning_frame()

    pos_p.pose.position.x += vectorx[0]*0.5
    pos_p.pose.position.y += vectorx[1]*0.5
    pos_p.pose.position.z += vectorx[2]*0.5
    scene.add_sphere(sphere_name,pos_p,radius = 0.03)

    return pos_p

def mark_traj(robot,group,scene,waypoints):
    i = 0
    for w in waypoints:
        pos_p = group.get_current_pose()
        pos_p.header.frame_id = robot.get_planning_frame()

        pos_p.pose.position.x = w.position.x
        pos_p.pose.position.y = w.position.y
        pos_p.pose.position.z = w.position.z
        scene.add_sphere("{}".format(i),pos_p,radius = 0.005)
        i = i+1

def remove_traj(scene):
    scene.remove_world_object()

def plot_motion_profiles(plan):
    position = []
    acceleration = []
    velocity = []
    duration = []
    for i in range(len(plan.joint_trajectory.points)):
        position.append(plan.joint_trajectory.points[i].positions)
        acceleration.append(plan.joint_trajectory.points[i].accelerations)
        velocity.append(plan.joint_trajectory.points[i].velocities)
        duration.append(plan.joint_trajectory.points[i].time_from_start.to_nsec()*1e-6)

    plt.title("position graph")
    plt.plot(duration, position) #position graph
    plt.show()

    plt.title("acceleration graph")
    plt.plot(duration, acceleration) #acceleration graph
    plt.show()

    plt.title("velocity graph")
    plt.plot(duration,velocity) #velocity graph
    plt.show()

    plt.title("duration")
    plt.plot(duration)
    plt.show()

def main():

    group,robot,scene = initialize_moveit("ur5_arm")
    rospy.sleep(3)

    pnt_pose = draw_sphere(group,scene,robot,"target")
    print("Target point is marked.")

    rospy.sleep(2)
    
    target = np.array([pnt_pose.pose.position.x,pnt_pose.pose.position.y,pnt_pose.pose.position.z])
    #_ = move_init_state(group)
    waypoints = b_spline_arcing(group, target, num_sampl_points=50, arcing_type=2, Print=True)
    print("Trajectory is plotted in a seperate figure.")
    
    rospy.sleep(2)
    mark_traj(robot,group,scene,waypoints)
    print("Trajectory is marked on the simulation.")
    rospy.sleep(2)
    remove_traj(scene)
    print("Trajectory markings are removed from simulation scene.")
    rospy.sleep(2)

    (plan, fraction) = group.compute_cartesian_path(
    waypoints, 0.01, 0.0) 
    for i in plan.joint_trajectory.points:
        if i.time_from_start == rospy.Duration(0):
            print("hi")
            plan.joint_trajectory.points.remove(i)

    for i in plan.joint_trajectory.points:
        print(i.time_from_start)
    

    print(fraction)
    rospy.sleep(2)

    group.execute(plan)
    print("Plan is being executed.")
    rospy.sleep(3)

    plot_motion_profiles(plan)


if __name__ == '__main__':
    try:      
        main()
    except rospy.ROSInterruptException:
        pass   