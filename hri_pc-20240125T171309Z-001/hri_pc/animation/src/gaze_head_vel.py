# Robot related libraries
import rospy
import moveit_commander
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray, String

from simple_pid import PID
from gaze_head_positional import gaze_vel
import copy
import time

global do_gaze

def button_callback(data):
    if data.data == "Gaze": 
        global do_gaze
        if do_gaze: do_gaze = False
        else: do_gaze = True

if __name__=="__main__":
    rospy.init_node("gazing_vel_based")
    group = moveit_commander.MoveGroupCommander("manipulator")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.Subscriber("button_topic", String, button_callback)
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=1)
    while not pub.get_num_connections(): rospy.sleep(0.1)
    
    r = 30  # Rate (Frequency)
    rate = rospy.Rate(r)
    # Kp, Ki, Kd = (5.0, 0.05, 0.1)
    Kp, Ki, Kd = (5.0, 0.05, 0.0)
    pid_w1 = PID(Kp, Ki, Kd, setpoint=0.0)
    pid_w2 = PID(Kp, Ki, Kd, setpoint=0.0)
    pid_w3 = PID(Kp, Ki, Kd, setpoint=0.0)
    
    do_gaze = False

    trans = tf_buffer.lookup_transform('base_link', 'flange', rospy.Time(0))
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
    b = int(input("period: "))
    
    # Breathing parameters: Amplitude, Frequency (bpm), Total Duration
    amplitude = 30.0
    bpm = 1.0/b
    i=1
    while not rospy.is_shutdown():
        if not do_gaze: 
            joint_vels = [0., 0., 0., 0., 0., 0.]
            # Publish joint vels to robot
            vel_msg = Float64MultiArray()
            vel_msg.data = joint_vels
            pub.publish(vel_msg)
            tick = time.time()
            gaze_started = False
            continue
        
        try:
            # if time.time() - tick > 30: do_gaze = False
            if time.time() - tick > 180: do_gaze = False
            if time.time() - tick > 5:
                joint_states = group.get_current_joint_values()
                transformation = tf_buffer.lookup_transform("base_link", "wrist_1_link", rospy.Time())
                rot = R.from_quat(np.array([transformation.transform.rotation.x,
                                        transformation.transform.rotation.y,
                                        transformation.transform.rotation.z,
                                        transformation.transform.rotation.w]))
                rot = rot.as_matrix()
                rot = np.vstack((rot, [0,0,0]))
                rot = np.hstack((rot, np.array([[transformation.transform.translation.x,
                                        transformation.transform.translation.y,
                                        transformation.transform.translation.z,
                                        1.0]]).T))
                gaze_point_pose = tf_buffer.lookup_transform("base_link", "helmet/base_link", rospy.Time())
                gaze_point = gaze_point_pose.transform.translation
                gaze_point = np.array([gaze_point.x, gaze_point.y, gaze_point.z, 1])
                gaze_point = gaze_point - [0.0, 0.08, 0.13, 0]
                
                desired_joints = np.array(gaze_vel(gaze_point, copy.deepcopy(joint_states), rot))
                
                pid_w1.setpoint = desired_joints[3]
                pid_w2.setpoint = desired_joints[4]
                pid_w3.setpoint = desired_joints[5]
                gaze_started = True
                joint_vels = [0., 0., 0., pid_w1(joint_states[3]), pid_w2(joint_states[4]), pid_w3(joint_states[5])]

            ##################
            # Breathing Part #
            ##################
            
            # Interpolate magnitude of velocity
            vel_mag = np.interp(num_of_vel_pts*bpm*i/r, indices, shape_vel)
            vel = breathe_vec * vel_mag * bpm * amplitude  # x = vt --> ax = (av)t
            i += 1 
            i = i%((r/bpm)+1)
            if i==0: i+=1
            joint_states = group.get_current_joint_values()
            jacobian = group.get_jacobian_matrix(joint_states)  # np array 6x6

            # rcond may be tuned not to get closer to singularities.
            # Take psuedo inverse of Jacobian.
            pinv_jacobian = np.linalg.pinv(jacobian, rcond=1e-15)  
            joint_vels = np.dot(pinv_jacobian[:3], vel)
            
            if gaze_started: joint_vels = np.concatenate((joint_vels, [pid_w1(joint_states[3]), pid_w2(joint_states[4]), pid_w3(joint_states[5])]))
            else: joint_vels = np.concatenate((joint_vels, [0., 0., 0.]))

            # Publish joint vels to robot
            vel_msg = Float64MultiArray()
            vel_msg.data = joint_vels
            pub.publish(vel_msg)
        except Exception as e:
            print(e)

        rate.sleep()