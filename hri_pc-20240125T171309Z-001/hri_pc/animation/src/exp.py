# Robot related libraries
import rospy
import moveit_commander
import numpy as np
import actionlib
import tf2_ros
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64MultiArray, String, Float64, Int16, Int16MultiArray
from controller_manager_msgs.srv import SwitchController
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

from simple_pid import PID
from gaze_head_positional import gaze_vel
import copy
import time
import yaml
import random 
import rosbag
import os

from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA


class StateMachine():
    def __init__(self, initial_state, set_of_states, time_to_ready = 1.0):
        self.state = initial_state
        self.initial_state = initial_state
        self.state_set = set_of_states
        self.tick = None
        self.time_trigger = False
        self.time_to_ready = time_to_ready

    def start_timer(self):
        if not self.time_trigger:
            self.tick = time.time()
            self.time_trigger = True
    
    def reset_timer(self):
        self.tick = None
        self.time_trigger = False   

    def set_state(self, state, time_dependent=False):
        if time_dependent: 
            self.reset_timer()
        if state in self.state_set:
            self.state = state
        else:
            rospy.logwarn("State is not in the set of states.")
        
    def reset(self):
        self.set_state(self.initial_state, True)


def connect_service(name):
    client = actionlib.SimpleActionClient(name, FollowJointTrajectoryAction)
    print(client.wait_for_server(timeout=rospy.Duration(1.0)))
    print("Connected to trajectory manager")
    return client

def reset_pids(Kp=1.0):
    # Kp, Ki, Kd = (5.0, 0.1, 0.1)
    # Kp, Ki, Kd = (3.0, 0.05, 0.0)
    Ki, Kd = (0.05, 0.0)
    pid_w1 = PID(Kp, Ki, Kd, setpoint=0.0)
    pid_w2 = PID(Kp, Ki, Kd, setpoint=0.0)
    pid_w3 = PID(Kp, Ki, Kd, setpoint=0.0)
    return [pid_w1, pid_w2, pid_w3]

def button_callback(data):
    if data.data == "Execute": 
        global run_exp
        if run_exp: run_exp = False
        else: run_exp = True

def next_callback(data):
    global flag
    global bag
    bag.write("intervine", String("next_button"))
    flag = True

def breathe_callback(data):
    global do_breathe
    if data.data == 2:
        do_breathe = True
    else:
        do_breathe = False

def gaze_callback(data):
    global do_gaze
    global Kp
    global pids
    if data.data == 2:
        do_gaze = True
        Kp = 3.0
        pids = reset_pids(Kp)
    else:
        do_gaze = False
        Kp = 3.0
        pids = reset_pids(Kp)

def ready_callback(data):
    global exp_id, sess_id
    exp_id, sess_id = data.data

def wait_wrist(tf_buffer, zone):
    try:
        transformation = tf_buffer.lookup_transform('world', 'human_wrist/base_link', rospy.Time(0))
        translation = transformation.transform.translation
        check_x_axis = translation.x > zone[0] and translation.x < (zone[0]+zone[2])
        check_y_axis = translation.y > zone[1] and translation.y < (zone[1]+zone[2])
        if check_y_axis and check_x_axis: return False
    except:
        return True
    return True

def gaze(group, tf_buffer, pids, target=None):
    joint_states = group.get_current_joint_values()
    try:
        transformation = tf_buffer.lookup_transform("world", "wrist_1_link", rospy.Time())
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
        if target is None:
            gaze_point_pose = tf_buffer.lookup_transform("world", "eye", rospy.Time())
            gaze_point = gaze_point_pose.transform.translation
            gaze_point = np.array([gaze_point.x, gaze_point.y, gaze_point.z, 1])
            # gaze_point = gaze_point - [0.0, 0.0, 0.15, 0]  # prev [0.0, 0.08, 0.13]  
        else:
            target = np.array(target)
            gaze_point = np.hstack((target, [0]))
        desired_joints = np.array(gaze_vel(gaze_point, copy.deepcopy(joint_states), rot))
    except Exception as e:
        desired_joints = np.array([0, 0, 0, joint_states[3], joint_states[4], joint_states[5]])    
    
    pids[0].setpoint = desired_joints[3]
    pids[1].setpoint = desired_joints[4]
    pids[2].setpoint = desired_joints[5]
    return np.array([pids[0](joint_states[3]), pids[1](joint_states[4]), pids[2](joint_states[5])])

def breathe(breathe_dict, i, gaze=True):
    shape_vel = breathe_dict["shape_vel"]
    num_of_vel_pts = breathe_dict["num_of_vel_pts"]
    indices = breathe_dict["indices"]
    amplitude = breathe_dict["amplitude"]
    bpm = breathe_dict["bpm"]
    breathe_vec = breathe_dict["breathe_vec"]
    r = breathe_dict["r"]

    # Interpolate magnitude of velocity
    vel_mag = np.interp(num_of_vel_pts*bpm*i/r, indices, shape_vel)
    vel = breathe_vec * vel_mag * bpm * amplitude  # x = vt --> ax = (av)t
    joint_states = group.get_current_joint_values()
    jacobian = group.get_jacobian_matrix(joint_states)  # np array 6x6

    # rcond may be tuned not to get closer to singularities.
    # Take psuedo inverse of Jacobian.
    pinv_jacobian = np.linalg.pinv(jacobian, rcond=1e-15)  
    if gaze: joint_vels = np.dot(pinv_jacobian[:3], vel)
    else: joint_vels = np.dot(pinv_jacobian, vel)
    return joint_vels

def create_traj(group, waypoints):
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = ["elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    
    for waypoint, tfs in waypoints:
        goal.trajectory.points.append(JointTrajectoryPoint())
        goal.trajectory.points[-1].positions = waypoint
        goal.trajectory.points[-1].velocities = [0., 0., 0., 0., 0., 0.]
        goal.trajectory.points[-1].accelerations = [0., 0., 0., 0., 0., 0.]  
        goal.trajectory.points[-1].effort = [0., 0., 0., 0., 0., 0.]  
        goal.trajectory.points[-1].time_from_start = rospy.Duration(float(tfs))

    goal.trajectory.header.stamp = rospy.Time.now()
    return goal

def gripper_callback(data):
    global  gripper_state
    gripper_state = data.data

def str_callback(msg):
    global flag
    if msg.data == "stop":
        flag = True

def deg2pi(deg):
    return np.pi * deg / 180.0

if __name__=="__main__":
    # Load the yaml file
    stream = open("exp.yaml", "r")
    dict = yaml.load(stream, Loader=yaml.Loader)

    flag = False

    # Initialization
    rospy.init_node("experiment")
    group = moveit_commander.MoveGroupCommander("manipulator")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, buff_size=120)
    rospy.Subscriber("button_topic", String, button_callback)
    rospy.Subscriber("gripper_state", Float64, gripper_callback)
    rospy.Subscriber("breathe_topic", Int16, breathe_callback)
    rospy.Subscriber("gaze_topic", Int16, gaze_callback)
    rospy.Subscriber("stopper", String, str_callback)
    rospy.Subscriber("ready_topic", Int16MultiArray, ready_callback)
    rospy.Subscriber("next_topic", String, next_callback)
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=1)
    gripper_pub = rospy.Publisher("gripper_control", String, queue_size=1)
    marker_pub = rospy.Publisher("marker", Marker, queue_size=1)
    # client = connect_service("vel_joint_traj_controller/follow_joint_trajectory")
    client = connect_service("pos_joint_traj_controller/follow_joint_trajectory")
    switch_controller = rospy.ServiceProxy(
                       '/controller_manager/switch_controller', SwitchController)

    while not pub.get_num_connections() and not gripper_pub.get_num_connections(): rospy.sleep(0.1)
    
    r = 30  # Rate (Frequency)
    rate = rospy.Rate(r)
    time.sleep(1)
    
    # Create experiment folder
    exp_id = -1
    sess_id = -1
    while exp_id < 0 and sess_id < 0:
        rate.sleep()

    dir = "/home/kovan-robot/animation_exp/{id}".format(id=exp_id)
    if not os.path.isdir(dir): os.mkdir(dir)
    bag_path = os.path.join(dir, "{sess_id}.bag".format(sess_id=sess_id))
    bag = rosbag.Bag(bag_path, "w")


    # Set breathing parameters
    breathe_dict = {}
    # trans = tf_buffer.lookup_transform('base_link', 'flange', rospy.Time(0))
    trans = tf_buffer.lookup_transform('base_link', 'world', rospy.Time(0))
    quat = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
    roti = R.from_quat(quat)
    # roti = np.dot(roti.as_matrix(), [-0.3,0.7,0])  # Breathe dir in ee frame
    roti = np.dot(roti.as_matrix(), [0,-0.6,0.4])  # Breathe dir in world frame
    breathe_vec = np.concatenate((roti, [0,0,0]))

    # Human data integrated, take difference
    shape_data = np.array([0.0, 0.005624999999999991, 0.014378906250000045, 
            0.024486547851562568, 0.03474537368774422, 0.0443933953943253, 
            0.0529963630275786, 0.060354717840215955, 0.06642930274659942, 
            0.07128390374926041, 0.07504226099316191, 0.0778570788273204, 
            0.07988866580429121, 0.08129106189085289, 0.08220379893995788, 
            0.08274774841852639, 0.08302380913885798, 0.0790451566321223, 
            0.07260624098380641, 0.06495160665441868, 0.05691987986583036, 
            0.04905405662967122, 0.041685214275588134, 0.039685214275588134, 0.03499542709050685, 
            0.02906453874530568, 0.02390450659228971, 0.019484261167040162, 
            0.015747394717907204, 0.012624483451303736, 0.010041439737111357, 
            0.007924965428885544, 0.006205920714800195, 0.004821221732756786, 
            0.0037147237823418333, 0.002837426374215357, 0.0021472441754255556, 
            0.0016085180924106934, 0.0011913883859058227, 0.000871112900045046, 
            0.0006273850763798272, 0.00044368593276999935])

    # shape_vel = np.array([shape_data[i+1]-shape_data[i] for i in range(len(shape_data)-1)])
    shape_vel = np.array([shape_data[i+1]-shape_data[i] for i in range(len(shape_data)-1)])
    shape_vel = np.hstack(([shape_data[-1]-shape_data[0]], shape_vel))
    # shape_vel = np.array(shape_vel)
    num_of_vel_pts = shape_vel.shape[0]
    indices = np.linspace(0, num_of_vel_pts-1, num_of_vel_pts)
    breathe_dict["shape_vel"] = shape_vel
    breathe_dict["num_of_vel_pts"] = num_of_vel_pts
    breathe_dict["indices"] = indices

    period = 4  # Period of a breathing
    # Breathing parameters: Amplitude, Frequency (bpm), Total Duration
    amplitude = 30.0
    bpm = 1.0/period
    i=1
    breathe_dict["amplitude"] = amplitude
    breathe_dict["bpm"] = bpm
    breathe_dict["breathe_vec"] = breathe_vec
    breathe_dict["r"] = r

    # Set experiment parameters
    state = "wait-wrist"
    do_gaze = False
    do_breathe = False
    run_exp = False
    gripper_state = None

    Kp = 3.0 if not do_gaze else 3.0
    pids = reset_pids(Kp)
    state_ind = 0
    story_line = dict["story-line"]
    button_position = dict["button-position"]
    loop_back = True
    nut = -1
    nut_pose = dict["bolt-poses"][nut]
    avail_nuts = dict["bolt-poses"]
    
    # Moving average setting for breathing
    moving_vels = []
    window_size = 20
    vel_cum = [0., 0., 0.]
    
    head_state = StateMachine(0, [0, 1, 2, 3, 4], 5.0)   # State 0 first come and gazing to nut
                                                # State 1 mutual gaze
                                                # State 2 cue towards nut
   
    wrist_state = StateMachine(0, [0, 1], 0.5)  # State 1 means request a new bolt, 
                                                # state 0 means take the bolt from table

    starting_mutual_gaze = None
    starting_mutual_gaze_flag = False
    end_exp_tick = None
    
    d = 0.45     # Diameter for virtual gazing sphere
                # Changes in the loop, check it

    nut_index = -1
    randomized_nuts = [0, 1, 2, 3]
    randomized_nuts = [3,2,1,0]
    randomized_nuts = [0,1,2,3]
    # random.shuffle(randomized_nuts)
    
    switch_controller(['pos_joint_traj_controller'],['joint_group_vel_controller'], 2, False, 1)

    home = copy.copy(dict["starting-pose"])
    trajectory = create_traj(group, home)
    client.send_goal(trajectory)
    result = client.wait_for_result(rospy.Duration(10.0))

    switch_controller(['joint_group_vel_controller'],['pos_joint_traj_controller'], 2, False, 1)
    
    bag.write("story_line", String(state))
    
    while not rospy.is_shutdown():
        if not run_exp: 
            # Publish joint vels to robot
            joint_vels = np.array([0., 0., 0., 0., 0., 0.])
            vel_msg = Float64MultiArray()
            vel_msg.data = joint_vels
            pub.publish(vel_msg)
            pids = reset_pids(Kp)
            continue


        if state == "wait-wrist":

            if wrist_state.state == 0 and not wait_wrist(tf_buffer, dict["wrist-zone"]):
                wrist_state.start_timer()
            elif wrist_state.state == 1:
                wrist_state.start_timer()
            else: wrist_state.reset_timer()

            if wrist_state.time_trigger and time.time() - wrist_state.tick > wrist_state.time_to_ready:
                if wrist_state.state == 0:
                    wrist_state.set_state(1, True)
                    wrist_state.start_timer()
                else:
                    wrist_state.set_state(0, True)
            if (wrist_state.state == 1 or flag):
                state_ind += 1
                flag = False
                state = story_line[state_ind]
                
                bag.write("story_line", String(state))
                
                if nut_index == 4:
                    vel_msg = Float64MultiArray()
                    vel_msg.data = [0, 0, 0, 0, 0, 0]
                    pub.publish(vel_msg)
                    rospy.sleep(0.1)
                    switch_controller(['pos_joint_traj_controller'],['joint_group_vel_controller'], 2, False, 1)

                    home = copy.copy(dict["starting-pose"])
                    trajectory = create_traj(group, home)
                    client.send_goal(trajectory)
                    result = client.wait_for_result(rospy.Duration(10.0))

                    switch_controller(['joint_group_vel_controller'],['pos_joint_traj_controller'], 2, False, 1)
                    
                    break  # End experiment

                # Switch to the trajectory controller
                nut = -1
                # Publish joint vels to robot
                vel_msg = Float64MultiArray()
                vel_msg.data = [0, 0, 0, 0, 0, 0]
                pub.publish(vel_msg)
                rospy.sleep(0.1)
                switch_controller(['pos_joint_traj_controller'],['joint_group_vel_controller'], 2, False, 1)
            
            if loop_back: 
                breathe_gaze_pt = tf_buffer.lookup_transform("world", "tool0_controller", rospy.Time())
                breathe_gaze_pt = [breathe_gaze_pt.transform.translation.x, breathe_gaze_pt.transform.translation.y, breathe_gaze_pt.transform.translation.z]
                head_state.set_state(0, True)
                head_state.time_to_ready = 4.0 # Changed at 22/02 2.5
                loop_back = False
                nut_index += 1
            
            if not end_exp_tick and nut_index == 4: end_exp_tick = time.time()
            if end_exp_tick and time.time() - end_exp_tick > 15.0: 
                vel_msg = Float64MultiArray()
                vel_msg.data = [0, 0, 0, 0, 0, 0]
                pub.publish(vel_msg)
                rospy.sleep(0.1)
                
                switch_controller(['pos_joint_traj_controller'],['joint_group_vel_controller'], 2, False, 1)

                home = copy.copy(dict["starting-pose"])
                trajectory = create_traj(group, home)
                client.send_goal(trajectory)
                result = client.wait_for_result(rospy.Duration(10.0))

                switch_controller(['joint_group_vel_controller'],['pos_joint_traj_controller'], 2, False, 1)
                
                break  # End experiment

            gaze_vels = np.array([0., 0., 0.])
            if do_gaze:
                try:
                    helmet_pose = tf_buffer.lookup_transform("world", "eye", rospy.Time())
                    helmet_pose_ori = helmet_pose.transform.rotation
                    helmet_p = helmet_pose.transform.translation
                    rot = R.from_quat(np.array([helmet_pose_ori.x, helmet_pose_ori.y, helmet_pose_ori.z, helmet_pose_ori.w ]))
                    rot = rot.as_matrix()
                    z = rot[:3, 1]
                    p = np.array([helmet_p.x, helmet_p.y, helmet_p.z])
                                        
                    cobot_head = tf_buffer.lookup_transform("world", "wrist_3_link", rospy.Time())
                    cobot_p = cobot_head.transform.translation
                    cobot_p_np = np.array([cobot_p.x, cobot_p.y, cobot_p.z])
                    b = np.dot(z, (p - cobot_p_np)) ** 2
                    ac = np.linalg.norm(p - cobot_p_np)**2 - (d/2)**2
                    delta = b - ac
                    if delta > 0:
                        color = ColorRGBA(0.0, 1.0, 0.0, 0.5)
                    else:
                        color = ColorRGBA(1.0, 0.0, 0.0, 0.5)

                    marker = Marker(
                        type=Marker.SPHERE,
                        id=1,
                        lifetime=rospy.Duration(10),
                        scale=Vector3(d, d, d),
                        header=Header(frame_id="world"),
                        pose= Pose(cobot_p, Quaternion(0, 0, 0, 1)),
                        color=color)
                    marker_pub.publish(marker)

                    if head_state.state == 0:
                        head_state.start_timer()
                        if time.time() - head_state.tick > head_state.time_to_ready:
                            if delta > 0:
                                head_state.set_state(3, True)
                                head_state.time_to_ready = 0.2
                                head_state.start_timer()
                    elif head_state.state == 3:
                        if not (delta > 0):
                            head_state.set_state(0, True)
                            head_state.start_timer()
                            head_state.time_to_ready = 0.2
                        elif time.time() - head_state.tick > head_state.time_to_ready:
                            head_state.set_state(1, True)
                            head_state.start_timer()
                            head_state.time_to_ready = 4.0
                    elif head_state.state == 1:
                        d = 0.65
                        if delta < 0 and time.time() - head_state.tick > 1.0:
                            head_state.set_state(0, True)
                            head_state.time_to_ready = 1.0
                            head_state.start_timer()
                            d = 0.55
                        elif time.time() - head_state.tick > head_state.time_to_ready:
                            head_state.set_state(4, True)
                            head_state.time_to_ready = 1.5
                            head_state.start_timer()
                    elif head_state.state == 2:
                        if time.time() - head_state.tick > head_state.time_to_ready:
                            head_state.set_state(4, True)
                            head_state.time_to_ready = 3.5
                            head_state.start_timer()
                    elif head_state.state == 4:
                        if time.time() - head_state.tick > head_state.time_to_ready:
                            head_state.set_state(1, True)
                            head_state.time_to_ready = 2.5
                            head_state.start_timer()

                    gaze_vels = gaze(group, tf_buffer, pids)
                    # if(head_state.state == 1) or nut == -1: gaze_vels = gaze(group, tf_buffer, pids)
                    # elif head_state.state == 4: gaze_vels = gaze(group, tf_buffer, pids, button_position)
                    # else: gaze_vels = gaze(group, tf_buffer, pids, nut_pose)
                except Exception as e:
                    print(e)
            else:
                try:
                    gaze_vels = gaze(group, tf_buffer, pids, breathe_gaze_pt)
                except Exception as e:
                    print(e)

            breathe_vels = np.array([0., 0., 0.])
            if do_breathe:
                try:
                    breathe_vels = breathe(breathe_dict, i, gaze=True)
                except Exception as e:
                    print(e)
                i += 1
                i = i%((r/bpm)+1)
                # if i==0: i+=1

            if len(moving_vels) == window_size:
                moving_vels.pop(0)
            moving_vels.append(breathe_vels)
            vel_cum = sum(moving_vels) / len(moving_vels)
            breathe_vels = vel_cum

            joint_vels = np.concatenate((breathe_vels, gaze_vels))

            # Publish joint vels to robot
            vel_msg = Float64MultiArray()
            vel_msg.data = joint_vels
            pub.publish(vel_msg)

            #####################
            # Bag required data #
            #####################
            bag.write("head_state", Int16(head_state.state))
            bag.write("wrist_state", Int16(wrist_state.state))

        elif state == "reach-bolt":
            if not do_gaze: trajectory = create_traj(group, dict["get-bolt-waypoints"])
            else: trajectory = create_traj(group, dict["gaze-get-bolt-waypoints"])
            client.send_goal(trajectory)
            result = client.wait_for_result(rospy.Duration(1.0/r))
            state = "on-action"

        # Wait the state to be done
        elif state == "on-action":
            if client.get_state() == 3:  # State 3 = SUCCEEDED
                state_ind += 1
                if len(story_line) == state_ind:
                    state_ind = 0
                    # Publish joint vels to robot
                    vel_msg = Float64MultiArray()
                    vel_msg.data = [0, 0, 0, 0, 0, 0]
                    pub.publish(vel_msg)
                    rospy.sleep(0.1)
                    switch_controller(['joint_group_vel_controller'],['pos_joint_traj_controller'], 2, False, 1)
                    rospy.sleep(0.1)
                    pids = reset_pids(Kp)
                    loop_back = True
                state = story_line[state_ind]
                bag.write("story_line", String(state))
        
        # Open the gripper
        elif state == "open-gripper":
            openness = 0.45
            gripper_pub.publish(String(str(openness)))
            state = "wait-gripper"
        
        elif state == "wait-gripper":
            openness_req = openness < 0.6 and gripper_state < 0.45 + 1e-2
            closeness_req = openness > 0.6 and gripper_state > 0.6 - 1e-2
            if gripper_state and (closeness_req or openness_req):
                state_ind += 1
                if len(story_line) == state_ind:
                    state_ind = 0
                    # Publish joint vels to robot
                    vel_msg = Float64MultiArray()
                    vel_msg.data = [0, 0, 0, 0, 0, 0]
                    rospy.sleep(0.1)
                    pub.publish(vel_msg)
                    switch_controller(['joint_group_vel_controller'],['pos_joint_traj_controller'], 2, False, 1)
                    rospy.sleep(0.1)
                    pids = reset_pids(Kp)
                    loop_back = True 
                state = story_line[state_ind]
                bag.write("story_line", String(state))


        # Grasp the bolt
        elif state == "grasp-bolt":
            grasp_pose = dict["grasp-pose"][nut_index]
            trajectory = create_traj(group, [grasp_pose])
            client.send_goal(trajectory)
            result = client.wait_for_result(rospy.Duration(1.0/r))
            state = "on-action"

        # Close the gripper
        elif state == "close-gripper":
            openness = 0.7
            gripper_pub.publish(String(str(openness)))
            state = "wait-gripper"
        
        # Bring the bolt
        elif state == "leave-bolt":
            trajectory = create_traj(group, dict["leave-bolt-waypoints"])
            client.send_goal(trajectory)
            result = client.wait_for_result(rospy.Duration(1.0/r))
            state = "on-action"

        elif state == "back-home":
            home = copy.copy(dict["home"])

            nut = randomized_nuts[nut_index]
            nut_pose = dict["bolt-poses"][nut]
            bag.write("selected_nut", Int16(nut))
            loop_back = False

            if not do_gaze:
                home.append(dict["stay-in-front"][nut])
            trajectory = create_traj(group, home)
            client.send_goal(trajectory)
            result = client.wait_for_result(rospy.Duration(1.0/r))
            state = "on-action"
        
        elif state == "pre-back-home":
            trajectory = create_traj(group, dict["pre-home"])
            client.send_goal(trajectory)
            result = client.wait_for_result(rospy.Duration(1.0/r))
            state = "on-action"
        
        rate.sleep()

    bag.close()
    print("Session ended")