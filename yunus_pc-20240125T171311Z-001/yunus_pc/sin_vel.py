import rospy
import moveit_commander
from std_msgs.msg import Float64MultiArray
import time
import numpy as np

if __name__=="__main__":

    rospy.init_node("BREATHER")
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=10) 
    while pub.get_num_connections() == 0:
        rospy.sleep(0.1)

    tick = time.time()
    r = 100
    rate = rospy.Rate(r)
    period = 5
    iterator = 0
    while time.time() - tick < 10:
        vel_msg = Float64MultiArray()
        joint_vel = np.sin(iterator*2*np.pi/(r*period)) / 5
        vel_msg.data = [0.0, 0.0, -0.02, 0., 0., 0.]
        pub.publish(vel_msg)
        iterator += 1
        rate.sleep()

    vel_msg.data = [0., 0., 0., 0., 0., 0.]
    pub.publish(vel_msg)
        