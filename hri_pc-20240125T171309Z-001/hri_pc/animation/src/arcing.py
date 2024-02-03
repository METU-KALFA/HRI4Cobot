""" This code is a wrapper for spline methods. 
It creates paths from spline methods and computes trajectories.

@author Burak Bolat
"""

# Robot related libraries
import rospy
import moveit_commander
import actionlib
from std_msgs.msg import String, Int16
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

# Spline libraries
# from scipy import interpolate
from b_spline import b_spline
from cr_spline import catmull_rom_2pt_in_3d
from bezier_spline import interpolate

import numpy as np

import utils
import copy

class Arcer:

    def __init__(self, **args):
        """ Check get_init_params()."""
        
        home_config = [1.57, 0.78, -1.80, -1.57, -1.57, -1.57]  # Predefined home position

        self.home_config = args["home_config"] if "home_config" in args else home_config

        # group_name is ur5_arm if simulation is launched with ur5_gripper package.
        # group_name is manipulator if simulation is launched with ur_gazebo package.
        group_name = args["group_name"] if "group_name" in args else "ur5_arm"     
        self.group, self.robot = self.init_moveit(group_name)

        self.interpolate_spline = args["interpolate_spline"] if "interpolate_spline" in args else b_spline
        self.time_parametrization = args["time_parametrization"] if "time_parametrization" in args else None

        self.plan = False
        self.traj = None
        self.execute = False
        rospy.Subscriber("button_topic", String, self.gui_subscriber)
        rospy.Subscriber("exagg_topic", Int16, self.exxag_callback)
        
        self.exxag_val = 0

    @staticmethod
    def get_init_params():
        """ Prints editable parameters."""
        
        print("home_config", "interpolate_spline", "time_parametrization", "group_name")

    def init_moveit(self, group_name):
        """ Initialize move_group with given group_name."""

        rospy.init_node('arcing')
        group = moveit_commander.MoveGroupCommander(group_name)
        robot = moveit_commander.RobotCommander()
        return group, robot

    def go_home(self):
        """ Moves the robot to the home position. Home position is defined when the class instance is created."""
        
        assert len(self.home_config) == 6
        self.group.go(self.home_config)

    def calculate_path(self, ctrl_pts, num_sampling):
        """ Calculates the path with given start and end points.
        Args:
            start (np.array): start point in format of (x,y,z,qx,qy,qz,qw)
            end (np.array)  : end point in format of (x,y,z,qx,qy,qz,qw)
            num_sampling (int)  : number of points that will be added waypoints list
        Returns:
            path (np.Array) : (num_of_waypoints, 3)
        """

        path = self.interpolate_spline(ctrl_pts, num_sampling)
        return path


    def calculate_trajectory(self, path):
        trajectory = self.time_parametrization(path)


    def gui_subscriber(self, data):
        print("Received button msg")
        if data.data == "Plan": self.plan = True
        elif data.data == "Execute": self.execute = True  

    def exxag_callback(self, data):
        self.exxag_val = data.data

def siso(trajectory):
    times = []
    # utils.plot_motion_profiles(trajectory, velocity=True)
    
    scalings = interpolate(np.array([[0,0], [0.33,0.6], [0.66,0.4], [1,1]]), len(trajectory.joint_trajectory.points), False)    

    for ind, point in enumerate(trajectory.joint_trajectory.points):
        # x = (ind/float(len(trajectory.joint_trajectory.points)))
        # scaling_ratio = (1 - ((2*x)-1)**2)
        # reset_time = scaling_ratio * trajectory.joint_trajectory.points[-1].time_from_start/2
        # if ind > len(trajectory.joint_trajectory.points)/2:
        #     reset_time = (1 + ((2*x)-1)**2) * trajectory.joint_trajectory.points[-1].time_from_start/2
        #     reset_time = point.time_from_start 
        reset_time = scalings[ind][1] * trajectory.joint_trajectory.points[-1].time_from_start
        point.time_from_start = reset_time
        # point.velocities = [y*scaling_ratio for y in point.velocities]
        point.velocities = []
        point.accelerations = []
            
    # utils.plot_motion_profiles(trajectory, velocity=True)
    return trajectory
        

if __name__=="__main__":

    mover = Arcer(group_name="manipulator", interpolate_spline=interpolate)
    mover.group.set_max_velocity_scaling_factor(0.1)
    mover.group.set_max_acceleration_scaling_factor(0.1)
    mover.go_home()
    rospy.sleep(1)
    
    # client = actionlib.SimpleActionClient("position_based_traj_controller/follow_joint_trajectory",
    #                                         FollowJointTrajectoryAction)
    client = actionlib.SimpleActionClient("vel_based_pos_traj_controller/follow_joint_trajectory",
                                            FollowJointTrajectoryAction)
    client.wait_for_server(timeout=rospy.Duration(1.0))
    start = utils.pose2np(copy.deepcopy(mover.group.get_current_pose().pose))[:3]
    drawer = utils.Drawer(start, start+np.array([0, 0.17, 0]), start+np.array([0.13, 0.13, 0]), start+np.array([0.1, 0.1, 0]))
    
    while not rospy.is_shutdown():
        start = drawer.start
        end = drawer.end
        c1 = drawer.control_1
        c2 = drawer.control_2
        exxagerate = 0.5
        c1 = c1 + (c1-start) * mover.exxag_val
        c2 = c2 + (c2-end) * mover.exxag_val
        path = mover.calculate_path(np.array([start[:3], c1, c2, end]), 50)
        drawer.visualize_path(path, [1, 0.4, 0, 1], diameter=0.05)

        if mover.plan:
            print("Started Planning")

            start = copy.deepcopy(mover.group.get_current_pose().pose)

            waypoints = []
            for point in path:
                wpose_next = copy.deepcopy(start)
                wpose_next.position.x = point[0]
                wpose_next.position.y = point[1]
                wpose_next.position.z = point[2]
                waypoints.append(wpose_next)
            
            trajectory, fraction = mover.group.compute_cartesian_path(waypoints, eef_step=0.01, jump_threshold=0)
            print(fraction)
            trajectory.joint_trajectory.points.remove(trajectory.joint_trajectory.points[0])
            mover.plan = False
            if fraction > 0.7:
                trajectory = mover.group.retime_trajectory(mover.robot.get_current_state(),
                                                            trajectory,
                                                            0.1,
                                                            0.05,
                                                            algorithm="time_optimal_trajectory_generation")
                # trajectory = siso(trajectory)
                mover.traj = trajectory

        if mover.execute:
            if mover.traj is None: print("No motion plan has been computed.")
            else: 
                goal = FollowJointTrajectoryGoal()
                goal.trajectory = copy.deepcopy(mover.traj.joint_trajectory)
                print(len(goal.trajectory.points))
                # client.send_goal(goal)
                mover.group.execute(mover.traj)
                mover.execute = False
                rospy.sleep(goal.trajectory.points[-1].time_from_start)
                mover.go_home()
                drawer.start = utils.pose2np(copy.deepcopy(mover.group.get_current_pose().pose))[:3]
