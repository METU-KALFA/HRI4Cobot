""" This code includes commonly used utilies 
    such as Rviz drawings, matplotlib drawings, velocity profile of trajectory plotting.

(add your name if you contribute this code)
@author Burak Bolat 
"""

import rospy
import tf.transformations as transformations
import numpy as np
import copy
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion
import matplotlib.pyplot as plt

# Drawing Libraries
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarkerControl, InteractiveMarkerFeedback
from std_msgs.msg import Header, ColorRGBA


# Interactive Marker Libraries
from interactive_markers.interactive_marker_server import *


class Drawer:
    
    def __init__(self, start, end, control_1, control_2):
        self.marker_arr_publisher = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
        
        self.server = InteractiveMarkerServer("spline_points")
        self.start = start
        self.end = end
        self.control_1 = control_1
        self.control_2 = control_2
        
        self.start_point = Point(self.start[0], self.start[1], self.start[2])
        self.end_point = Point(self.end[0], self.end[1], self.end[2])
        self.control_1_point = Point(self.control_1[0], self.control_1[1], self.control_1[2])
        self.control_2_point = Point(self.control_2[0], self.control_2[1], self.control_2[2])
        
        # self.make_6Dof_marker(True, InteractiveMarkerControl.NONE, self.start_point, False, "start")
        self.make_6Dof_marker(False, InteractiveMarkerControl.MOVE_3D, self.end_point, False, "end", scale=0.2)
        self.make_6Dof_marker(False, InteractiveMarkerControl.MOVE_3D, self.control_1_point, False, "c1", scale=0.2)
        self.make_6Dof_marker(False, InteractiveMarkerControl.MOVE_3D, self.control_2_point, False, "c2", scale=0.2)
        self.server.applyChanges()

        # Wait for the publisher to be ready
        while self.marker_arr_publisher.get_num_connections() == 0: pass

    def create_sphere(self, id, s, d, frame_id, color):
        
        start = Point()

        start.x = s[0]
        start.y = s[1]
        start.z = s[2]

        marker = Marker(
                    type=Marker.SPHERE,
                    id=id,
                    lifetime=rospy.Duration(10),
                    scale=Vector3(d, d, d),
                    header=Header(frame_id=frame_id),
                    pose= Pose(Point(start.x, start.y, start.z), Quaternion(0, 0, 0, 1)),
                    color=ColorRGBA(color[0], color[1], color[2], color[3]))

        return marker

    def create_line(self, id, waypoints, scale, frame_id, color):
        marker = Marker(
            type=Marker.LINE_STRIP,
            id=id,
            lifetime=rospy.Duration(0.01),
            scale=Vector3(scale, 1, 1),
            header=Header(frame_id=frame_id),
            pose= Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
            color=ColorRGBA(color[0], color[1], color[2], color[3])
        )

        for point in waypoints:
            marker.points.append(Point(point[0], point[1], point[2]))

        return marker

    def visualize_path(self, waypoints, color=np.array([0., 1., 0., 1.]), diameter=0.15, frame="world"):
        """
        Args:
            color (np.Array): [r,g,b,a] 
        """

        path_marker_arr = MarkerArray()
        # marker = self.create_line(len(waypoints)+1, waypoints, diameter, "world", color)
        # path_marker_arr.markers.append(copy.deepcopy(marker))

        for id, waypoint in enumerate(waypoints):
            marker = self.create_sphere(id, waypoint, diameter, frame, color)
            path_marker_arr.markers.append(copy.deepcopy(marker))

        # while self.marker_arr_publisher.get_num_connections() == 0:
        #     rospy.spin()
        self.marker_arr_publisher.publish(path_marker_arr)


    def processFeedback(self, feedback):
        s = "Feedback from marker '" + feedback.marker_name
        s += "' /control '" + feedback.control_name + "'"

        mp = ""
        if feedback.mouse_point_valid:
            mp = " at " + str(feedback.mouse_point.x)
            mp += ", " + str(feedback.mouse_point.y)
            mp += ", " + str(feedback.mouse_point.z)
            mp += " in frame " + feedback.header.frame_id

        # if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
        #     rospy.loginfo( s + ": button click" + mp + "." )
        # elif feedback.event_type == InteractiveMarkerFeedback.MENU_SELECT:
        #     rospy.loginfo( s + ": menu item " + str(feedback.menu_entry_id) + " clicked" + mp + "." )
        # elif feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
        #     rospy.loginfo( s + ": pose changed")
        # elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_DOWN:
        #     rospy.loginfo( s + ": mouse down" + mp + "." )
        # elif feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
        #     rospy.loginfo( s + ": mouse up" + mp + "." )

        if feedback.marker_name == "start":
            self.start = np.array([feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z])
        elif feedback.marker_name == "end":
            self.end = np.array([feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z])
        elif feedback.marker_name == "c1":
            self.control_1 = np.array([feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z])
        elif feedback.marker_name == "c2":
            self.control_2 = np.array([feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z])
            
        self.server.applyChanges()


    def makeSphere(self, msg):
        marker = Marker()

        marker.type = Marker.SPHERE
        marker.scale.x = msg.scale * 0.45
        marker.scale.y = msg.scale * 0.45
        marker.scale.z = msg.scale * 0.45
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0

        return marker


    def make_sphere_control(self, msg):
        control =  InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.makeSphere(msg))
        msg.controls.append(control)

        return control


    def make_6Dof_marker(self, fixed, interaction_mode, position, show_6dof=False, marker_name="No_name", scale=1):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "world"
        int_marker.pose.position = position
        int_marker.pose.orientation = Quaternion(0, 0, 0, 1)
        int_marker.scale = scale

        int_marker.name = marker_name

        # insert a box
        self.make_sphere_control(int_marker)
        int_marker.controls[0].interaction_mode = interaction_mode

        
        int_marker.description = marker_name 

        if interaction_mode != InteractiveMarkerControl.NONE:
            control_modes_dict = { 
                            InteractiveMarkerControl.MOVE_3D : "MOVE_3D",
                            InteractiveMarkerControl.ROTATE_3D : "ROTATE_3D",
                            InteractiveMarkerControl.MOVE_ROTATE_3D : "MOVE_ROTATE_3D" }
        
        if show_6dof: 
            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            control.name = "rotate_x"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            control.name = "move_x"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            control.name = "rotate_z"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            control.name = "move_z"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.name = "rotate_y"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.name = "move_y"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            if fixed:
                control.orientation_mode = InteractiveMarkerControl.FIXED
            int_marker.controls.append(control)

        self.server.insert(int_marker, self.processFeedback)

def pose2quad(pose):
    return np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])


def pose2np(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z, 
                    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

def np_pose2ros_pose(pose):
    temp_pose = Pose()
    temp_pose.position.x, temp_pose.position.y, temp_pose.position.z, temp_pose.orientation.x, temp_pose.orientation.y, temp_pose.orientation.z, temp_pose.orientation.w = pose

    return temp_pose

def get_gripper_vector(current_pose, dir_gripper):
    w2ee_mat = transformations.quaternion_matrix(pose2quad(current_pose))
    ee2w_mat = transformations.inverse_matrix(w2ee_mat)
    # Calculate direction vector representation in worl frame
    dir_world = np.matmul(dir_gripper, ee2w_mat)  # Check the order of vector
    dir_world = transformations.unit_vector(dir_world[:3])
    return dir_world


def plot_motion_profiles(plan, position=False, velocity=False, acceleration=False, 
                        joint_list = [True, True, True, True, True, True]):
    assert len(joint_list) == 6
    
    positions = []
    accelerations = []
    velocities = []
    durations = []

    for point in plan.joint_trajectory.points:
        if position: positions.append(np.array(point.positions)[joint_list])
        if velocity: velocities.append(np.array(point.velocities)[joint_list])
        if acceleration: accelerations.append(np.array(point.accelerations)[joint_list])
        durations.append(point.time_from_start.to_nsec()*1e-9)

    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    durations = np.array(durations)
    
    if position:
        plt.title("position graph")
        plt.plot(durations, positions)  # position graph
        plt.show()

    if velocity:
        plt.title("velocity graph")
        plt.plot(durations,velocities)  # velocity graph
        plt.show()
    
    if acceleration:
        plt.title("acceleration graph")
        plt.plot(durations, accelerations)  # acceleration graph
        plt.show()