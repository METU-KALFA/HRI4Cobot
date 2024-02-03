import numpy as np
import time
import rospy
import moveit_commander
import tf2_ros

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

import breathing
import gazing

joint_states_global = {}
def js_callback(data):
    global joint_states_global
    joint_states_global["pos"] = np.array([data.position[2], 
                                  data.position[1], 
                                  data.position[0], 
                                  data.position[3], 
                                  data.position[4], 
                                  data.position[5]])
    
    joint_states_global["vels"] = np.array([data.velocity[2], 
                                  data.velocity[1], 
                                  data.velocity[0], 
                                  data.velocity[3], 
                                  data.velocity[4], 
                                  data.velocity[5]])
    
if __name__ == "__main__":
    rospy.init_node("hri4cobot_maincontroller")
    group = moveit_commander.MoveGroupCommander("manipulator")  # move_group for getting jacobian, any other jacobian lib is fine
    
    # Get joint states
    rospy.Subscriber("joint_states", JointState, js_callback)
    
    # Find the topic for ros_control to take velocity command
    joint_vel_command_topic = "joint_group_vel_controller/command"
    joint_vel_publisher = rospy.Publisher(joint_vel_command_topic, Float64MultiArray, queue_size=1)
    while not joint_vel_publisher.get_num_connections(): rospy.sleep(0.1)

    control_rate = 200  # Frequency for main control loop
    ros_rate = rospy.Rate(control_rate)  
    
    # Set breathing parameters 
    breathe_dict = {}
    breathe_dict["control_rate"] = control_rate

    # Set breathe vector direction in base_link frame
    breathe_vec = np.array([-1, 1, 0.5])  # named as r in the paper
    breathe_vec = breathe_vec / np.linalg.norm(breathe_vec) if np.all(breathe_vec > 0) else breathe_vec
    breathe_dict["breathe_vec"] = breathe_vec

    # Human breathing data based breathing velocity profile function f(·)
    human_data_path = "breathe_data.npy"
    f = np.load(human_data_path)
    breathe_dict["f"] = f

    period = 3 # Time in seconds for a breathe
    freq = 1 / period  # Named as beta in the paper
    amplitude = 0.5  # Tune this. It changes wrt. control rate
    breathe_dict["freq"] = freq
    breathe_dict["amplitude"] = amplitude

    num_of_total_joints = 6  # UR5 has 6 joints
    # Use at least 3 joints for breathing, below that, it is not effective
    num_of_breathing_joints = 3  # Num of body joints starting from base towards to end effector
    breathe_dict["num_of_joints"] = num_of_breathing_joints

    breathe_dict["compensation"] = True
    breathe_controller = breathing.Breather(breathe_dict)

    # Set gazing parameters
    gaze_dict = {}
    gaze_dict["kp"] = 5.0
    gaze_dict["kd"] = 0.0
    gaze_dict["ki"] = 0.05

    while not rospy.is_shutdown() and breathe_controller.breathe_count < 2:
        loop_start_time = rospy.Time.now().to_sec()
        velocity_command = breathe_controller.step(joint_states_global["pos"],
                                                    joint_states_global["vels"],
                                                    group.get_jacobian_matrix)
        # Publish joint vels to robot
        velocity_command = np.concatenate((velocity_command, [0]*(num_of_total_joints - num_of_breathing_joints)))
        vel_msg = Float64MultiArray()
        vel_msg.data = velocity_command.tolist()
        joint_vel_publisher.publish(vel_msg)
        ros_rate.sleep()

    # Zero joint vels of robot before exiting
    # Publish joint vels to robot
    vel_msg = Float64MultiArray()
    vel_msg.data = [0.0] * 6
    joint_vel_publisher.publish(vel_msg)
    rospy.sleep(1.0)
    