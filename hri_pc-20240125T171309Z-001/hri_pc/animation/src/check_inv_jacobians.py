import rospy
import time
import numpy as np
import moveit_commander

import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray

def main():
    group = moveit_commander.MoveGroupCommander("manipulator")
    rospy.init_node("check_inv_jacobian")
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=10) 
    while pub.get_num_connections() == 0:
        rospy.sleep(0.1)
    

    r = 30
    rate = rospy.Rate(r)

    preds = []
    measurements = []
    loop_var = 0  # Measures how many loop are done to see if code has bottleneck

    # Set velocity of end effector
    vel = np.array([0., 0.1, 0., 0., 0., 0.])  # [x, y, z, alpha, beta, gamma] = [linear, angular]
    start = rospy.get_time()
    
    # Add starting points
    # Read pose and joint states once. First readings take larger time that breaks expected loop structure.
    joint_states = group.get_current_joint_values()
    pose = group.get_current_pose()
    preds.append([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
    measurements.append([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])

    while not rospy.is_shutdown() and rospy.get_time() - start < 2:
        loop_var += 1
        joint_states = group.get_current_joint_values()
        pose = group.get_current_pose()
        jacobian = group.get_jacobian_matrix(joint_states)  # np array 6x6
        
        # rcond may be tuned not to get closer to singularities.
        # Take psuedo inverse of Jacobian.
        pinv_jacobian = np.linalg.pinv(jacobian, rcond=1e-15)  
        joint_vels = np.dot(pinv_jacobian, vel)
        position = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        pred_position = position + (vel[:3]/float(r))

        # Publish joint vels to robot
        vel_msg = Float64MultiArray()
        vel_msg.data = joint_vels.tolist()
        pub.publish(vel_msg)

        # Wait for time period
        rate.sleep()

        
        # Measure the pose and save data
        pose_after = group.get_current_pose()
        preds.append(pred_position)
        measurements.append([pose_after.pose.position.x, pose_after.pose.position.y, pose_after.pose.position.z])        
        
    # Stop the robot
    vel = np.array([0., 0., 0., 0., 0., 0.])
    vel_msg = Float64MultiArray()
    vel_msg.data = vel.tolist()
    pub.publish(vel_msg)
    # Plot Pred vs Measured
    fig, ax = plt.subplots()
    ax.plot(np.array(preds)[:,1], label="preds")
    ax.plot(np.array(measurements)[:,1], label="measures")
    ax.legend(loc="upper right")
    ax.set_ylabel('y of ee (m)')
    ax.set_xlabel('time (s/rate)')
    ax.set_title('Position check of Inv Jacobian. Rate={}\n Total dist: {}, Expected: {}'.format(r, measurements[-1][1]-measurements[0][1], loop_var/float(r)*0.1))
    plt.show()
    # plt.savefig("/home/kovan-beta/animation_materials/inv_jac_{}_r.png".format(r))

if __name__=="__main__": 
    main()