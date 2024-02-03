import rospy
import time
import numpy as np
import moveit_commander

from std_msgs.msg import Float64MultiArray

def main():
    group = moveit_commander.MoveGroupCommander("manipulator")
    rospy.init_node("check_inv_jacobian")
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=10) 
    while pub.get_num_connections() == 0:
        rospy.sleep(0.1)

    r = 30  # Rate
    rate = rospy.Rate(r)

    # Read pose and joint states once. First readings take larger time that breaks expected loop structure.
    joint_states = group.get_current_joint_values()
    pose = group.get_current_pose()

    # Human data
    shape_data = np.array([0.0, 0.005624999999999991, 0.014378906250000045, 
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

    # Breathing parameters: Amplitude, Frequency (bpm), Total Duration, Direction
    # Implemented in trajectory.py
    breathe_vec = np.array([0, 0, 1, 0, 0, 0])
    amplitude = 1.0
    bpm = 1.0

    i=0

    while not rospy.is_shutdown():

        vel = breathe_vec * shape_data[i]
        i += 1 % shape_data.shape[0]
        
        joint_states = group.get_current_joint_values()
        jacobian = group.get_jacobian_matrix(joint_states)  # np array 6x6

        # rcond may be tuned not to get closer to singularities.
        # Take psuedo inverse of Jacobian.
        pinv_jacobian = np.linalg.pinv(jacobian, rcond=1e-15)  
        joint_vels = np.dot(pinv_jacobian, vel)
        

if __name__=="__main__": main()