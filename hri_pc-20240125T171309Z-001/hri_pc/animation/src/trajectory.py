# import roslib
import rospy
import actionlib
import moveit_commander
import time
import numpy as np

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray, String
import tf2_ros
import tf.transformations as tft
from scipy.spatial.transform import Rotation as R

from controller_manager_msgs.srv import SwitchController

import matplotlib.pyplot as plt

def connect_service(name):
    client = actionlib.SimpleActionClient(name,
                                            FollowJointTrajectoryAction)
    print(client.wait_for_server(timeout=rospy.Duration(1.0)))
    print("Connected to trajectory manager")
    return client

def switch_controller_mode(switch_controller, client, mode):
    # Mode 0: velocity, Mode 1: trajectory
    if mode == 0:
        ret = switch_controller(['joint_group_vel_controller'], 
                           ['scaled_pos_joint_traj_controller'], 2, False, 1)
    elif mode == 1:
        ret = switch_controller(['scaled_pos_joint_traj_controller'],
                            ['joint_group_vel_controller'], 2, False, 1)

    return ret

def str_callback(msg):
    global flag
    if msg.data == "stop":
        flag = True

def main():
    group = moveit_commander.MoveGroupCommander("manipulator")
    rospy.init_node("trajectory_controll")

    rospy.wait_for_service('/controller_manager/switch_controller')
    print("Controller manager is alive.")    
    switch_controller = rospy.ServiceProxy(
                       '/controller_manager/switch_controller', SwitchController)
    
    
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    client = connect_service("scaled_pos_joint_traj_controller/follow_joint_trajectory")
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=10) 
    rospy.Subscriber("stopper", String, str_callback)
    while pub.get_num_connections() == 0:
        rospy.sleep(0.1)

    r = 30  # Rate (Frequency)
    rate = rospy.Rate(r)

    while True:
        if flag: 
            vel_msg = Float64MultiArray()
            vel_msg.data = [0., 0., 0., 0., 0., 0.]
            pub.publish(vel_msg)
            rospy.sleep(0.1)
            exit()
        # a = input("num: ")
        a = 2.0

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.points.append(JointTrajectoryPoint())
        goal.trajectory.points.append(JointTrajectoryPoint())
        goal.trajectory.points.append(JointTrajectoryPoint())
        goal.trajectory.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        
        # goal.trajectory.points[0].positions = current
        # goal.trajectory.points[1].positions = [1.57, 0.9, -2.1, -1.57, -1.57, -1.57]  # Elbow Down
        goal.trajectory.points[0].positions = [1.7, -0.9, 2.1, -4.0, -1.57, -1.57]  # Elbow Up Posture 1
        goal.trajectory.points[0].velocities = [0, 0., 0., 0., 0., 0.]  # Elbow Up Posture 1
        goal.trajectory.points[0].accelerations = [0., 0., 0., 0., 0., 0.]  # Elbow Up Posture 1
        goal.trajectory.points[0].effort = [0., 0., 0., 0., 0., 0.]  # Elbow Up Posture 1
        
        goal.trajectory.points[1].positions = [1.07, -0.9, 2.1, -4.0, -1.57, -1.57]  # Elbow Up Posture 1
        goal.trajectory.points[1].velocities = [0., 0., 0., 0., 0., 0.]  # Elbow Up Posture 1
        goal.trajectory.points[1].accelerations = [0., 0., 0., 0., 0., 0.]  # Elbow Up Posture 1
        goal.trajectory.points[1].effort = [0., 0., 0., 0., 0., 0.]  # Elbow Up Posture 1
        
        goal.trajectory.points[2].positions = [2.4, -0.9, 2.1, -4.0, -1.57, -1.57]  # Elbow Up Posture 1
        goal.trajectory.points[2].velocities = [0., 0., 0., 0., 0., 0.]  # Elbow Up Posture 1
        goal.trajectory.points[2].accelerations = [0., 0., 0., 0., 0., 0.]  # Elbow Up Posture 1
        goal.trajectory.points[2].effort = [0., 0., 0., 0., 0., 0.]  # Elbow Up Posture 1
        
        goal.trajectory.header.stamp = rospy.Time.now()# + rospy.Duration(1)
        goal.trajectory.points[0].time_from_start = rospy.Duration(float(a))
        goal.trajectory.points[1].time_from_start = rospy.Duration(1.5*float(a))
        goal.trajectory.points[2].time_from_start = rospy.Duration(2.5*float(a))
        
        client.send_goal(goal)
        print("the goal is sent")
        # Wait for up to 5 seconds for the motion to complete
        result = client.wait_for_result(rospy.Duration(100.0))
        print(result)
        print("-----------------------")
        print("Switching controller to vel")
        # print(switch_controller_mode(switch_controller, client, 0))

        vel_msg = Float64MultiArray()
        
        # Read pose and joint states once. First readings take larger time that breaks expected loop structure.
        joint_states = group.get_current_joint_values()
        
        # !!! Comment following code !!!
        
        trans = tfBuffer.lookup_transform('base_link', 'flange', rospy.Time(0))
        quat = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
        roti = R.from_quat(quat)
        roti = np.dot(roti.as_matrix(), [0.3,0.7,0])  # Breathe dir in ee frame
        breathe_vec = np.concatenate((roti, [0,0,0]))
        
        # Human data integrated, take difference
        shape_data = np.array([0.0, 0., 0.005624999999999991, 0.014378906250000045, 
                0.024486547851562568, 0.03474537368774422, 0.0443933953943253, 
                0.0529963630275786, 0.060354717840215955, 0.06642930274659942, 
                0.07128390374926041, 0.07504226099316191, 0.0778570788273204, 
                0.07988866580429121, 0.08129106189085289, 0.08220379893995788, 
                0.08274774841852639, 0.08302380913885798, 0.0790451566321223, 
                0.07260624098380641, 0.06495160665441868, 0.05691987986583036, 
                0.04905405662967122, 0.041685214275588134, 0.03499542709050685, 
                0.02906453874530568, 0.02390450659228971, 0.019484261167040162, 
                0.015747394717907204, 0.012624483451303736, 0.010041439737111357, 
                0.007924965428885544, 0.006205920714800195, 0.004821221732756786, 
                0.0037147237823418333, 0.002837426374215357, 0.0021472441754255556, 
                0.0016085180924106934, 0.0011913883859058227, 0.000871112900045046, 
                0.0006273850763798272, 0.00044368593276999935])
        shape_vel = np.array([shape_data[i+1]-shape_data[i] for i in range(len(shape_data)-1)])
        num_of_vel_pts = shape_vel.shape[0]
        indices = np.linspace(1, num_of_vel_pts, num_of_vel_pts)
        
        print("Period >= 3 feels good")
        # b = int(input("period: "))
        b = 3
        
        # Breathing parameters: Amplitude, Frequency (bpm), Total Duration
        amplitude = 30.0
        bpm = 1.0/b
        i=1

        # start = rospy.get_time()
        # while rospy.get_time() - start < 5:
        #     # Interpolate magnitude of velocity
        #     vel_mag = np.interp(num_of_vel_pts*bpm*i/r, indices, shape_vel)
        #     vel = breathe_vec * vel_mag * bpm * amplitude  # x = vt --> ax = (av)t
        #     i += 1 
        #     i = i%((r/bpm)+1)
        #     if i==0: i+=1
        #     joint_states = group.get_current_joint_values()
        #     jacobian = group.get_jacobian_matrix(joint_states)  # np array 6x6

        #     # rcond may be tuned not to get closer to singularities.
        #     # Take psuedo inverse of Jacobian.
        #     pinv_jacobian = np.linalg.pinv(jacobian, rcond=1e-15)  
        #     joint_vels = np.dot(pinv_jacobian[:3], vel)
        #     joint_vels = np.concatenate((joint_vels, [0,0,0]))
        #     # Publish joint vels to robot
        #     vel_msg = Float64MultiArray()
        #     vel_msg.data = joint_vels.tolist()
        #     pub.publish(vel_msg)

        #     rate.sleep()

        # vel_msg = Float64MultiArray()
        # vel_msg.data = [0., 0., 0., 0., 0., 0.]
        # pub.publish(vel_msg)
        # print("Switching controller to traj")
        print(switch_controller_mode(switch_controller, client, 1))

if __name__ == "__main__": 
    flag = False
    main()
