import rospy
import time
import numpy as np
import moveit_commander

import matplotlib.pyplot as plt

from std_msgs.msg import Float64MultiArray

def main(): 
    rospy.init_node("check_jacobian")
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=10) 
    while pub.get_num_connections() == 0:
        rospy.sleep(0.1)

    group = moveit_commander.MoveGroupCommander("manipulator")
    r = 120
    rate = rospy.Rate(r)

    preds = []
    measurements = []
    
    # Set velocity of joints
    vel = np.array([0., 0., -0.5, 0., 0., 0.])
    vel_msg = Float64MultiArray()
    vel_msg.data = vel.tolist()
    pub.publish(vel_msg)
    
    start = rospy.get_time()
    while not rospy.is_shutdown() and rospy.get_time() - start < 3:
        joint_states = group.get_current_joint_values()
        pose = group.get_current_pose()
        jacobian = group.get_jacobian_matrix(joint_states)  # np array 6x6
        
        # Make prediction using time period and Jacobian 
        pred_vel = np.dot(jacobian, vel.T) / float(r)
        position = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        pred_position = pred_vel[:3] + position
        
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

    # np.save("preds", np.array(preds))
    # np.save("measurements", np.array(measurements))
    
    # Plot Pred vs Measured
    fig, ax = plt.subplots()
    ax.plot(np.array(preds)[:,2], label="preds")
    ax.plot(np.array(measurements)[:,2], label="measures")
    ax.legend(loc="upper left")
    ax.set_ylabel('z of ee (m)')
    ax.set_xlabel('time (s/rate)')
    ax.set_title('Position check of Jacobian. Rate={}\nJoint:3 Vel:0.5 rad/s'.format(r))
    plt.show()  

if __name__ == "__main__": main()