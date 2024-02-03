import rospy
import moveit_commander
import tf2_ros
from std_msgs.msg import Float64MultiArray
import numpy as np
from sensor_msgs.msg import JointState
import copy

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

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

class Compensator():
    def __init__(self) -> None:
        self.interrupt_index, self.exit_index = None, None
        self.exit_pos = None
        self.exit_vel = None
        self.n_joints = 6
        self.ff = np.repeat([1], self.n_joints)
        self.kp, self.kd = np.repeat([5], self.n_joints), np.repeat([0.1], self.n_joints)
        self.error, self.d_error = np.zeros(self.n_joints), np.zeros(self.n_joints)
        self.error_matrix = np.zeros((self.n_joints,1)) 
        
    def compensate(self):
        global joint_states_global
        error = self.exit_pos - joint_states_global["pos"]
        d_error = self.exit_vel - joint_states_global["vels"]
        self.error_matrix = np.concatenate((self.error_matrix, error.reshape(6,1)), axis=1)
        command = self.exit_vel*self.ff + error*self.kp + d_error*self.kd
        return command

if __name__ == "__main__":
    rospy.init_node("traj_control")

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    rospy.Subscriber("joint_states", JointState, js_callback)
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=10) 
    while pub.get_num_connections() == 0:
        rospy.sleep(0.1)

    rate = rospy.Rate(200)

    comp = Compensator()
    # comp.exit_pos = copy.deepcopy(joint_states_global["pos"])
    # comp.exit_pos[1] += 0.15
    # comp.exit_vel = np.zeros_like(comp.exit_pos)
    # comp.exit_vel[1] = 0.0
    x = [0, 2]
    y = [joint_states_global["pos"][1], joint_states_global["pos"][1]-0.15]
    cs = CubicSpline(x, y, bc_type=((1, 0), (1, 0.0)))
    xs = np.arange(x[0], x[1], rate.sleep_dur.nsecs*1e-9)
    vals = cs(xs)
    speeds = cs(xs, 1)

    i = 0
    while not rospy.is_shutdown():
        comp.exit_pos = copy.deepcopy(joint_states_global["pos"])  
        comp.exit_pos[1] = vals[i]
        comp.exit_vel = np.zeros_like(comp.exit_pos)
        comp.exit_vel[1] = speeds[i]

        # if np.all(np.abs(comp.exit_pos - joint_states_global["pos"]) < 1e-5):
        #     print("Ended", joint_states_global["pos"], joint_states_global["vels"])
        #     # Publish joint vels to robot
        #     vel_msg = Float64MultiArray()
        #     vel_msg.data = [0.0] * 6
        #     pub.publish(vel_msg)
        #     rospy.sleep(0.01)
        #     exit()

        if i == len(vals) - 1:
            print("Ended", joint_states_global["pos"], joint_states_global["vels"])
            vel_msg = Float64MultiArray()
            vel_msg.data = [0.0] * 6
            pub.publish(vel_msg)
            rospy.sleep(0.1)
            exit()

        command = comp.compensate()

        # Publish joint vels to robot
        vel_msg = Float64MultiArray()
        vel_msg.data = command.tolist()
        pub.publish(vel_msg)
        i += 1
        rate.sleep()
