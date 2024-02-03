# Robot related libraries
import rospy
import moveit_commander
import numpy as np
import tf2_ros
import copy
import time
import rosbag
import json
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from scipy.interpolate import CubicSpline
from math import sqrt
joint_states_global = {}
config = {
    "exp_number" : "exp014",
    "interrupt_index" : 0.45,   
    "exit_index" : 0.60,
    "amplitude" : 30.0/80.0,
    "period" : 3,
    "rate" : 200,
    "compansete_after": 1,
    "exp_time" : 600.0,
    "breathe_vector" : [sqrt(2),sqrt(2),0] # This is in flange frame
}

with open("/home/kovan3/ur5_ws/src/Exps/" + config["exp_number"] + "/config.json", "w") as f:
    json.dump(config, f)

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
    try:
        bag.write('joint_states',data)
    except:
        pass
class Compensator():
    def __init__(self) -> None:
        self.interrupt_index, self.exit_index, self.time_to_recover  = None, None, None
        self.exit_pos, self.target_pos = None, None
        self.exit_vel, self.target_vel = None, None
        self.n_joints = 6
        self.ff = np.repeat([1], self.n_joints)
        self.kp, self.kd = np.repeat([5], self.n_joints), np.repeat([0.1], self.n_joints)
        self.error, self.d_error = np.zeros(self.n_joints), np.zeros(self.n_joints)
        self.error_matrix = np.zeros((self.n_joints,1)) 
        
    def generate_segment(self, rate):
        global joint_states_global
        self.i = 0
        x = [0, self.time_to_recover]
        
        y_arr = [[y_init, y_target] for y_init, y_target in zip(joint_states_global["pos"], self.target_pos)]
        bc_arr = [[v_init, v_target] for v_init, v_target in zip(joint_states_global["vels"], self.target_vel)]
        cs_arr = [CubicSpline(x, y, bc_type=((1, bc[0]), (1, bc[1]))) for y, bc in zip(y_arr, bc_arr)]
        xs = np.arange(x[0], x[1], rate.sleep_dur.nsecs*1e-9)
        self.vals = [cs(xs) for cs in cs_arr]
        self.speeds = [cs(xs, 1) for cs in cs_arr]
        return self.vals, self.speeds

    def reset(self):
        self.i = 0
        self.exit_pos = None
        self.exit_vel = None
        self.error = np.zeros(self.n_joints)
        self.d_error = np.zeros(self.n_joints)

    def compensate(self):
        global joint_states_global
        des_pos = np.array([vals[self.i] for vals in self.vals])
        des_vel = np.array([vals[self.i] for vals in self.speeds])
        error = des_pos - joint_states_global["pos"]
        d_error = des_vel - joint_states_global["vels"]
        self.error_matrix = np.concatenate((self.error_matrix, error.reshape(6,1)), axis=1)
        command = des_vel*self.ff + error*self.kp + d_error*self.kd
        self.i += 1
        return command
    
if __name__== "__main__":    
    rospy.init_node("breather")
    group = moveit_commander.MoveGroupCommander("manipulator")

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    rospy.Subscriber("joint_states", JointState, js_callback)
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=10) 
    while pub.get_num_connections() == 0:
        rospy.sleep(0.1)

    pos = []
    vel_msg = Float64MultiArray()
    r = 200 # Rate (Frequency)
    rate = rospy.Rate(r)

    # Read pose and joint states once. First readings take larger time that breaks expected loop structure.
    joint_states = group.get_current_joint_values()
    
    trans = tfBuffer.lookup_transform('base_link', 'flange', rospy.Time(0))
    quat = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
    roti = R.from_quat(quat)
    roti = roti.as_matrix()
    roti = np.dot(roti, config["breathe_vector"])  # Breathe dir in ee frame
    roti = roti / np.linalg.norm(roti)
    breathe_vec = np.concatenate((roti, [0,0,0]))
    # breathe_vec = np.array([0, 0, 1, 0, 0, 0])

    shape_vel = np.load("/home/kovan3/ur5_ws/src/breathe_data_1.npy")
    num_of_vel_pts = shape_vel.shape[0]
    indices = np.linspace(1, num_of_vel_pts, num_of_vel_pts)
    print("Period >= 3 feels good")
    period = config["period"]
    
    # Breathing parameters: Amplitude, Frequency (bps), Total Duration
    amplitude = config["amplitude"]
    bps = 1.0/period
    i = 0

    breathe_body = False

    comp = Compensator()
    comp.interrupt_index = int(config["interrupt_index"] * int(r/bps))  # At %65 of the breathing, take the control
    comp.exit_index = int(config["exit_index"] * int(r/bps))  # At %80 of the breathing, leave the control
    bag = rosbag.Bag('/home/kovan3/ur5_ws/src/Exps/' + config["exp_number"] + '/joint_states_bag', 'w')

    first_breathe = True
    first_loop_comp = True
    compensate_after = config["compansete_after"]  # After number - 1 breathe
    breathe_count = 0
    exit_positions = []
    exit_vels = []
    input_positions = []
    input_vels = []
    tick = time.time()
    while not rospy.is_shutdown() and time.time() - tick < config["exp_time"]:
        joint_states = group.get_current_joint_values()

        if i == comp.exit_index:
            exit_positions.append(copy.deepcopy(joint_states_global["pos"]))
            exit_vels.append(copy.deepcopy(joint_states_global["vels"]))

        if i == comp.interrupt_index:
            input_positions.append(copy.deepcopy(joint_states_global["pos"]))
            input_vels.append(copy.deepcopy(joint_states_global["vels"]))
            
        if first_breathe and i == comp.interrupt_index:
            starting_time = time.time()

        if first_breathe and i == comp.exit_index:
            comp.time_to_recover = time.time() - starting_time
            comp.target_pos = copy.deepcopy(joint_states_global["pos"])
            comp.target_vel = copy.deepcopy(joint_states_global["vels"])
            first_breathe = False
        
        joint_vels = [0.0] * 6
        if breathe_count > 0 \
                and breathe_count % compensate_after == compensate_after-1 \
                and i < comp.exit_index \
                and i > comp.interrupt_index:

            
            if first_loop_comp:
                vals, speeds = comp.generate_segment(rate)  # Generates cubic spline
                # vel_msg = Float64MultiArray()
                # vel_msg.data = [0.0] * 6
                # pub.publish(vel_msg)
                # rospy.sleep(0.1)
                # import matplotlib.pyplot as plt
                # plt.plot(vals[0])
                # plt.show()
                # plt.plot(speeds[0])
                # plt.show()

                first_loop_comp = False

            joint_vels = comp.compensate()            
            if breathe_body: joint_vels[3:] = [0,0,0]
        else:
            if not first_loop_comp:
                first_loop_comp = True
                comp.reset()

            vel_mag = np.interp(num_of_vel_pts*bps*i/r, indices, shape_vel)
            vel = breathe_vec * vel_mag * bps * amplitude  # x = vt --> ax = (av)t

            jacobian = group.get_jacobian_matrix(joint_states)  # np array 6x6
            # rcond may be tuned not to get closer to singularities.
            # Take psuedo inverse of Jacobian.
            pinv_jacobian = np.linalg.pinv(jacobian, rcond=1e-15)  

            if breathe_body:
                joint_vels = np.dot(pinv_jacobian[:3], vel)
                joint_vels = np.concatenate((joint_vels, [0,0,0]))
            else:
                joint_vels = np.dot(pinv_jacobian, vel)
        


        # Publish joint vels to robot
        vel_msg = Float64MultiArray()
        vel_msg.data = joint_vels.tolist()
        pub.publish(vel_msg)

        i = i + 1 
        i = i % int(r/bps)
        if i==0: breathe_count += 1

        rate.sleep()

    # Publish joint vels to robot
    vel_msg = Float64MultiArray()
    vel_msg.data = [0.0] * 6
    pub.publish(vel_msg)
    rospy.sleep(1.0)
    exit_positions = np.array(exit_positions)
    exit_vels = np.array(exit_vels)
    input_positions = np.array(input_positions)
    input_vels = np.array(input_vels)
    np.save("/home/kovan3/ur5_ws/src/Exps/" + config["exp_number"] + "/exit_positions", exit_positions)
    np.save("/home/kovan3/ur5_ws/src/Exps/" + config["exp_number"] + "/exit_vels", exit_vels)
    np.save("/home/kovan3/ur5_ws/src/Exps/" + config["exp_number"] + "/input_positions", input_positions)
    np.save("/home/kovan3/ur5_ws/src/Exps/" + config["exp_number"] + "/input_vels", input_vels)
    bag.close()