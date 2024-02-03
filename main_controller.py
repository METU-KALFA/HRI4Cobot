import numpy as np
import time
import rospy
import moveit_commander
import tf2_ros
from scipy.spatial.transform import Rotation as R

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
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, buff_size=120)
    
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
    gaze_dict["kp"] = 1.0
    gaze_dict["kd"] = 0.0
    gaze_dict["ki"] = 0.05
    # Initial guesses are the initial joint values of the robot starting from the initial head joint
    # In ur wrist_1, wrist_2, wrist_3 are the head joints
    gaze_dict["initial_guesses"] = [-1.57, -1.57, -3.14]  # Decide accounting your robots initial joint values and gazing area
    gazing_controller = gazing.Gazer(gaze_dict)

    do_breathing = False
    do_gazing = True

    while not rospy.is_shutdown() and breathe_controller.breathe_count < 2:
        loop_start_time = rospy.Time.now().to_sec()
        if do_breathing:
            breathing_velocities = breathe_controller.step(joint_states_global["pos"],
                                                    joint_states_global["vels"],
                                                    group.get_jacobian_matrix)
        
        if do_gazing:
            # Calculate head transformation matrix
            # !!! Change "base_link" with "world" if the gazing is in the world frame !!!
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
            gazing_target = np.array([0, 1, 0])  # !!! Change this wrt. the gazing target, gazing in base_lin frame !!!
            gazing_velocities = gazing_controller.step(gazing_target, r, joint_states_global["pos"])
        
        # Publish joint vels to robot
        if do_breathing and not do_gazing: velocity_command = np.concatenate((breathing_velocities, [0]*(num_of_total_joints - num_of_breathing_joints)))
        elif do_gazing and not do_breathing: velocity_command = np.concatenate(([0]*3, gazing_velocities))
        else: velocity_command = np.concatenate((breathing_velocities[:3], gazing_velocities))

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
    