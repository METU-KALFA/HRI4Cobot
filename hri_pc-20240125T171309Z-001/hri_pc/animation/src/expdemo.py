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
import os

from geometry_msgs.msg import Pose, Point, Vector3, Quaternion, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import random

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

def current_gaze_err(tf_buffer, target_gaze):
    trf2 = tf_buffer.lookup_transform("world", "wrist_2_link", rospy.Time())
    trf3 = tf_buffer.lookup_transform("world", "wrist_3_link", rospy.Time())
    trf2_pos = np.array([trf2.transform.translation.x, trf2.transform.translation.y, trf2.transform.translation.z])
    
    x = trf3.transform.translation.x - trf2.transform.translation.x
    y = trf3.transform.translation.y - trf2.transform.translation.y
    z = trf3.transform.translation.z - trf2.transform.translation.z
    
    curr_gaze = np.array([x,y,z])
    tar_gaze_vec = target_gaze - trf2_pos
    tar_gaze_vec /= np.linalg.norm(tar_gaze_vec)
    curr_gaze_vec = curr_gaze
    curr_gaze_vec /= np.linalg.norm(curr_gaze_vec)
    cos_theta = np.inner(curr_gaze_vec, tar_gaze_vec)
    theta = np.rad2deg(np.arccos(cos_theta))
    return theta
    

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
        else:
            target = np.array(target)
            gaze_point = np.hstack((target, [0]))
        desired_joints = np.array(gaze_vel(gaze_point, copy.deepcopy(joint_states), rot))
    except Exception as e:
        desired_joints = np.array([0, 0, 0, joint_states[3], joint_states[4], joint_states[5]])    
    
    pids[0].setpoint = desired_joints[3]
    pids[1].setpoint = desired_joints[4]
    pids[2].setpoint = desired_joints[5]
    gaze_vels = np.array([pids[0](joint_states[3]), pids[1](joint_states[4]), pids[2](joint_states[5])])
    gaze_vels = gaze_vels.clip(-0.9, 0.9)
    return gaze_vels

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

def create_traj(waypoints):
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

def task_callback(data):
    global tasking_flag
    tasking_flag = True

def str_callback(msg):
    global flag
    if msg.data == "stop":
        flag = True

def deg2pi(deg):
    return np.pi * deg / 180.0

if __name__=="__main__":
    # Initialization
    rospy.init_node("experiment")
    group = moveit_commander.MoveGroupCommander("manipulator")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer, buff_size=120)
    
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=1)
    gripper_pub = rospy.Publisher("gripper_control", String, queue_size=1)
    marker_pub = rospy.Publisher("marker", Marker, queue_size=1)
    client = connect_service("pos_joint_traj_controller/follow_joint_trajectory")
    switch_controller = rospy.ServiceProxy(
                       '/controller_manager/switch_controller', SwitchController)
    rospy.Subscriber("gripper_state", Float64, gripper_callback)
    rospy.Subscriber("tasking_state", Float64, task_callback)
    while not pub.get_num_connections() and not gripper_pub.get_num_connections(): rospy.sleep(0.1)
    
    r = 30  # Rate (Frequency)
    rate = rospy.Rate(r)
    time.sleep(1)
    

    # Set breathing parameters
    breathe_dict = {}
    trans = tf_buffer.lookup_transform('base_link', 'world', rospy.Time(0))
    quat = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
    roti = R.from_quat(quat)
    # roti = np.dot(roti.as_matrix(), [-0.3,0.7,0])  # Breathe dir in ee frame
    roti = np.dot(roti.as_matrix(), [-0.6,0,0.4])  # Breathe dir in world frame
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
    amplitude = 60.0
    bpm = 1.0/period
    i=1
    breathe_dict["amplitude"] = amplitude
    breathe_dict["bpm"] = bpm
    breathe_dict["breathe_vec"] = breathe_vec
    breathe_dict["r"] = r

    Kp = 8.0
    pids = None
    pids2 = None
    
    # Moving average setting for breathing
    moving_vels = []
    window_size = 20
    vel_cum = [0., 0., 0.]
   
    wrist_state = StateMachine(0, [0, 1], 0.5)  # State 1 means request a new bolt, 
                                                # state 0 means take the bolt from table


    story_ind = 0
    story_arr = ["self_play", "first_interaction", "leave_part", "indicate_task", "tasking", "thank", "gaze_drop", "gift", "indicate_task2", "after_gift", "thank2", "home_last"]
    
    # Poses for keyframes
    initial_pose = [[[2.0175061225891113, -1.3720091024981897, 1.2277528047561646, -3.8543227354632776, -1.6718218962298792, -3.126575295125143], 2]]
    leave = [
        [[0.9070968627929688, -1.0169161001788538, 0.7688401341438293, -1.5150240103351038, -1.5294807592975062, -3.1831546465503138], 2],
        [[0.9136567115783691, -0.8022273222552698, 0.7688880562782288, -1.620049778615133, -1.5300796667682093, -3.183178488408224], 3]
    ]

    after_leave1 = [[[0.9121246337890625, -0.9238475004779261, 0.7687082886695862, -1.6066706816302698, -1.5274913946734827, -3.1832144896136683], 1]]
    after_leave2 = [
        [[1.028576374053955, -0.9621217886554163, 0.7811644077301025, -1.5751922766314905, -1.5275152365313929, -3.183190647755758], 0.5],
        [[1.1880168914794922, -0.9657319227801722, 0.7777923345565796, -1.7604387442218226, -1.5339153448687952, -3.1820407549487513], 1.2],
        [[0.4655933380126953, -0.5469616095172327, 0.8017690181732178, -1.5773604551898401, -1.5403507391559046, -3.2198105494128626], 3]
    ]

    metu_logo_world = [0.75907, -0.24334, 0.78485]
    base_part_world = [0.87839, -0.65875, 0.7037]

    home = [[[2.1719870567321777, -1.4838269392596644, 0.6962235569953918, -3.872636620198385, -1.4640167395221155, -3.1266232172595423], 3]]
    home_last = [[[2.043973922729492, -1.3904183546649378, 0.6748613119125366, -3.772507969533102, -1.4599297682391565, -3.12632400194277], 3],
                 [[2.043973922729492, -1.3904183546649378, 0.6748613119125366, -3.772507969533102, -1.4599297682391565, -2.8], 3.5]
                 ]

    take_obj = [
        [[1.7890548706054688, -1.5326522032367151, -0.8126462141620081, -2.107375446950094, -1.509573761616842, -3.133089367543355], 3],
        [[1.6497960090637207, -1.2751429716693323, -0.8846047560321253, -2.0261319319354456, -1.5024784247027796, -3.3199575583087366], 4]
        
    ]

    leave_obj = [
        [[1.7890548706054688, -1.5326522032367151, -0.8126462141620081, -2.107375446950094, -1.509573761616842, -3.133089367543355], 1],
        [[1.925187587738037, -1.1886900107013147, 1.2882386445999146, -3.0331814924823206, -2.151252571736471, -3.720464293156759],4],
        [[1.905782699584961, -1.268111530934469, 1.052937388420105, -2.9363730589496058, -1.617563549672262, -2.746965233479635],5]
    ]

    play_keys = [
        [[1.5038304328918457, -1.1246212164508265, 1.2477829456329346, -2.6220067183123987, -1.381585423146383, -3.1264317671405237], 1],
        [[1.495349407196045, -1.0843589941607874, 1.3924063444137573, -2.5630229155169886, -1.6592519919024866, -2.6572497526751917], 2],
        [[0.9370975494384766, -0.8038337866412562, 1.3518011569976807, -2.04533559480776, -1.5486558119403284, -2.6576924959765833], 3]
    ]

    over_gift = [[[1.344118595123291, -1.295722786580221, 2.1710824966430664, -1.7618878523456019, -1.5093939940081995, -3.347768847142355], 4]]
    take_gift = [[[1.3461666107177734, -1.0874179045306605, 2.1710705757141113, -1.7898147741900843, -1.5097296873675745, -3.34778076807131], 1]]
    left_gift = [[[1.8955116271972656, -1.5728734175311487, 0.7265856266021729, -2.4908483664142054, -1.514571491871969, -3.284062449132101], 4],
                 [[1.372683048248291, -1.0544756094561976, 0.7184610366821289, -2.1071837584124964, -1.5100768248187464, -3.2822421232806605], 5]]
    
    last_thanking_pose = [[[1.7235441207885742, -1.2541921774493616, 0.6601596474647522, -3.523369614277975, -1.0561469236956995, -1.723501984273092],1]]

    switch_controller(['pos_joint_traj_controller'],['joint_group_vel_controller'], 2, False, 1)
    trajectory = create_traj(initial_pose)
    client.send_goal(trajectory)
    result = client.wait_for_result(rospy.Duration(10.0))
    controller = "pos"
    
    tick = time.time()
    wrist_1 = wrist_2 = wrist_3 = 0
    helmet_timer = None     # To stop self_play after helmet seen
    intro_timer = None      # To stop introductionary mutual gazing
    indicate_timer = None   # To stop indicating the task
    tasking_timer = None    # To mutual gaze with experimenter

    indicate_ind = -1  # 0: human, 1:upper part, 2: base part
    indicate_inner_timer = None
    ind_inner_time = 1.5  # Change attention in indicating task. 
    time_offset = 0.0     # Care indicate_time is dividable by ind_inner_time * 3 + time_offset * 2

    helmet_seen_time = 1
    intro_play_time = 5
    indicate_time = 5
    tasking_time = 3

    thank_up = None
    thank_down = None
    thank_mid = None
    thank_ind = 0
    total_nod = 0

    nod_1 = np.sin(np.linspace(-np.pi,0,15))
    nod_2 = np.sin(np.linspace(0,np.pi, 30))
    nod_3 = np.sin(np.linspace(0, -np.pi,30))
    nod_5 = np.sin(np.linspace(0,np.pi, 15))

    nod = np.concatenate((nod_1, nod_2, nod_3, nod_5))
    nod_length = nod.shape[0]

    nod_i = 0 

    closeness_req = False
    openness_req = False
    g_state = None
    finish = False
    flag_leave_part = None
    tasking_flag = None

    gripper_action = "idle"
    while not rospy.is_shutdown():
        if gripper_action and gripper_action != "idle":
            if gripper_action == "open":
                openness = 0.25
                gripper_pub.publish(String(str(openness)))
                gripper_action = "wait_action"
            elif gripper_action == "close":
                openness = 0.6
                gripper_pub.publish(String(str(openness)))
                gripper_action = "wait_action"
            
            if gripper_action == "wait_action":
                openness_req = openness < 0.55 and gripper_state < 0.25 + 1e-2
                closeness_req = openness >= 0.55 and gripper_state > 0.55 - 1e-2
            
            if openness_req or closeness_req:
                openness_req = False
                closeness_req = False
                gripper_action = "idle"

        elif story_arr[story_ind] == "self_play":
            trajectory = create_traj(play_keys)
            client.send_goal(trajectory)
            result = client.wait_for_result(rospy.Duration(10.0))
            
            # wrist_1 = (wrist_1 + (random.random() - 0.5) / 5) / 2 # 2 lere basma
            # wrist_2 = (wrist_2 + (random.random() - 0.5) / 5) / 2 
            # wrist_3 = (wrist_3 + (random.random() -0.5)) / 2

            # vel_msg = Float64MultiArray()
            # vel_msg.data = [0., 0., 0., wrist_1, wrist_2, wrist_3]
            # pub.publish(vel_msg)    

            if not helmet_timer and tf_buffer.can_transform("world", "helmet/base_link", rospy.Time()):
                print("Helmet seen")
                helmet_timer = time.time()

            if helmet_timer and time.time() - helmet_timer > helmet_seen_time:
                # trajectory = create_traj(home)
                # client.send_goal(trajectory)
                # result = client.wait_for_result(rospy.Duration(10.0))
            
                story_ind += 1               
                switch_controller(['joint_group_vel_controller'],['pos_joint_traj_controller'], 2, False, 1)
                controller = "vel"

        elif story_arr[story_ind] == "first_interaction":
            if not intro_timer:
                intro_timer = time.time()
            
            try:
                if not pids: pids = reset_pids(Kp)
                gaze_vels = gaze(group, tf_buffer, pids)
                breathe_vels = breathe(breathe_dict, i, gaze=True)
                i += 1
                i = i%((r/bpm)+1)

                if len(moving_vels) == window_size:
                    moving_vels.pop(0)
                moving_vels.append(breathe_vels)
                vel_cum = sum(moving_vels) / len(moving_vels)
                breathe_vels = vel_cum

                joint_vels = np.concatenate((breathe_vels, gaze_vels))

                vel_msg = Float64MultiArray()
                vel_msg.data = joint_vels
                pub.publish(vel_msg)

                if intro_timer and time.time() - intro_timer > intro_play_time:
                    vel_msg = Float64MultiArray()
                    vel_msg.data = [0., 0., 0., 0., 0., 0.]
                    pub.publish(vel_msg)
                    story_ind += 1
                    i = 0
                    moving_vels = []    
                               

            except Exception as e:
                print(e)
                vel_msg = Float64MultiArray()
                vel_msg.data = [0., 0., 0., 0., 0., 0.]
                pub.publish(vel_msg)

        elif story_arr[story_ind] == "leave_part":
            if not controller == "pos":
                result = switch_controller(['pos_joint_traj_controller'], ['joint_group_vel_controller'], 2, False, 1)
                controller = "pos" if result else "vel"
            
            if not flag_leave_part:
                trajectory = create_traj(leave)
                client.send_goal(trajectory)
                result = client.wait_for_result(rospy.Duration(10.0))
                flag_leave_part = "second"
                gripper_action = "open"

            elif flag_leave_part == "second":
                trajectory = create_traj(after_leave1)
                client.send_goal(trajectory)
                result = client.wait_for_result(rospy.Duration(10.0))
                flag_leave_part = "third"
                gripper_action = "close"

            elif flag_leave_part == "third":
                trajectory = create_traj(after_leave2)
                client.send_goal(trajectory)
                result = client.wait_for_result(rospy.Duration(10.0))
                flag_leave_part = "fourth"
            
            elif flag_leave_part == "fourth":
                trajectory = create_traj(home)
                client.send_goal(trajectory)
                result = client.wait_for_result(rospy.Duration(10.0))
                flag_leave_part = None
                story_ind += 1

        elif story_arr[story_ind] == "indicate_task":
            if not controller == "vel":
                switch_controller(['joint_group_vel_controller'], ['pos_joint_traj_controller'], 2, False, 1)
                controller = "vel" if result else "pos"
    
            if not pids2: pids2 = reset_pids(3)

            if not indicate_timer:
                indicate_timer = time.time()
            
            try:
                if not indicate_inner_timer:
                    indicate_inner_timer = time.time()
                    indicate_ind += 1 
                    indicate_ind %= 2

                if indicate_ind == 0:
                    gaze_vels = gaze(group, tf_buffer, pids2)
                elif indicate_ind == 1:
                    gaze_vels = gaze(group, tf_buffer, pids2, base_part_world)                    
                
                breathe_vels = breathe(breathe_dict, i, gaze=True)
                i += 1
                i = i%((r/bpm)+1)

                if len(moving_vels) == window_size:
                    moving_vels.pop(0)
                moving_vels.append(breathe_vels)
                vel_cum = sum(moving_vels) / len(moving_vels)
                breathe_vels = vel_cum

                joint_vels = np.concatenate((breathe_vels, gaze_vels))


                if indicate_inner_timer and \
                    (indicate_ind == 0 and time.time() - indicate_inner_timer > ind_inner_time) or \
                    (indicate_ind != 0 and time.time() - indicate_inner_timer > ind_inner_time + time_offset):
                    indicate_inner_timer = None

                vel_msg = Float64MultiArray()
                vel_msg.data = joint_vels
                pub.publish(vel_msg)

                if tasking_flag:
                    indicate_inner_timer = None
                    indicate_ind = 0
                    vel_msg = Float64MultiArray()
                    vel_msg.data = [0., 0., 0., 0., 0., 0.]
                    pub.publish(vel_msg)
                    story_ind += 1    
                    i = 0
                    moving_vels = []
                    tasking_flag = False

            except Exception as e:
                print(e)
                vel_msg = Float64MultiArray()
                vel_msg.data = [0., 0., 0., 0., 0., 0.]
                pub.publish(vel_msg)
                pids2 = None

        elif story_arr[story_ind] == "tasking":
            if not pids2: pids2 = reset_pids(1)

            try:
                gaze_vels = gaze(group, tf_buffer, pids2)
                breathe_vels = breathe(breathe_dict, i, gaze=True)
                i += 1
                i = i%((r/bpm)+1)

                if len(moving_vels) == window_size:
                    moving_vels.pop(0)
                moving_vels.append(breathe_vels)
                vel_cum = sum(moving_vels) / len(moving_vels)
                breathe_vels = vel_cum

                joint_vels = np.concatenate((breathe_vels, gaze_vels))

                vel_msg = Float64MultiArray()
                vel_msg.data = joint_vels

                pub.publish(vel_msg)
            
                if tasking_flag:
                        vel_msg = Float64MultiArray()
                        vel_msg.data = [0., 0., 0., 0., 0., 0.]
                        pub.publish(vel_msg)
                        story_ind += 1 
                        i = 0
                        moving_vels = []
                        tasking_flag = False
                        pids2 = None

            except Exception as e:
                print(e)
                vel_msg = Float64MultiArray()
                vel_msg.data = [0., 0., 0., 0., 0., 0.]
                pub.publish(vel_msg)

        elif story_arr[story_ind] == "thank":
            
            vel_msg = Float64MultiArray()
            joint_velocities = np.array([0, 0, 0, nod[nod_i], 0, 0])
            vel_msg.data = joint_velocities
            pub.publish(vel_msg)

            nod_i += 1
            nod_i = nod_i % nod_length
            if nod_i == 0:
                vel_msg.data = np.array([0, 0, 0, 0, 0, 0])
                pub.publish(vel_msg)
                rate.sleep()
                story_ind += 1    

        elif story_arr[story_ind] == "gaze_drop":
            
            if not pids2: pids2 = reset_pids(2)

            try:
                if not indicate_inner_timer:
                    indicate_inner_timer = time.time()
                    indicate_ind += 1 
                    indicate_ind %= 2

                if indicate_ind == 0:
                    gaze_vels = gaze(group, tf_buffer, pids2)
                elif indicate_ind == 1:
                    gaze_vels = gaze(group, tf_buffer, pids2, base_part_world)
                
                breathe_vels = breathe(breathe_dict, i, gaze=True)
                i += 1
                i = i%((r/bpm)+1)

                if len(moving_vels) == window_size:
                    moving_vels.pop(0)
                moving_vels.append(breathe_vels)
                vel_cum = sum(moving_vels) / len(moving_vels)
                breathe_vels = vel_cum

                joint_vels = np.concatenate((breathe_vels, gaze_vels))


                if indicate_inner_timer and \
                    (indicate_ind == 0 and time.time() - indicate_inner_timer > ind_inner_time) or \
                    (indicate_ind != 0 and time.time() - indicate_inner_timer > ind_inner_time + time_offset):
                    indicate_inner_timer = None

                vel_msg = Float64MultiArray()
                vel_msg.data = joint_vels
                pub.publish(vel_msg)

                if tasking_flag:
                    indicate_inner_timer = None
                    indicate_ind = 0
                    vel_msg = Float64MultiArray()
                    vel_msg.data = [0., 0., 0., 0., 0., 0.]
                    pub.publish(vel_msg)
                    story_ind += 1    
                    i = 0
                    moving_vels = []
                    pids2 = None
                    tasking_flag = False

            except Exception as e:
                print(e)
                vel_msg = Float64MultiArray()
                vel_msg.data = [0., 0., 0., 0., 0., 0.]
                pub.publish(vel_msg)

        elif story_arr[story_ind] == "gift":
            if not controller == "pos":
                result = switch_controller(['pos_joint_traj_controller'], ['joint_group_vel_controller'], 2, False, 1)
                controller = "pos" if result else "vel"
            
            if not flag_leave_part:
                trajectory = create_traj(over_gift)
                client.send_goal(trajectory)
                result = client.wait_for_result(rospy.Duration(10.0))
                flag_leave_part = "second"
                gripper_action = "open"

            elif flag_leave_part == "second":
                trajectory = create_traj(take_gift)
                client.send_goal(trajectory)
                result = client.wait_for_result(rospy.Duration(10.0))
                flag_leave_part = "third"
                gripper_action = "close"

            elif flag_leave_part == "third":
                trajectory = create_traj(left_gift)
                client.send_goal(trajectory)
                result = client.wait_for_result(rospy.Duration(10.0))
                flag_leave_part = "fourth"
                gripper_action = "open"

            elif flag_leave_part == "fourth":
                trajectory = create_traj(home)
                client.send_goal(trajectory)
                result = client.wait_for_result(rospy.Duration(10.0))
                gripper_action = "close"
                flag_leave_part = "fifth"
                story_ind += 1
                tasking_flag = False

        elif story_arr[story_ind] == "indicate_task2":
            if not controller == "vel":
                switch_controller(['joint_group_vel_controller'], ['pos_joint_traj_controller'], 2, False, 1)
                controller = "vel" if result else "pos"
    
            if not pids2: pids2 = reset_pids(3)

            if not indicate_timer:
                indicate_timer = time.time()
            
            try:
                if not indicate_inner_timer:
                    indicate_inner_timer = time.time()
                    indicate_ind += 1 
                    indicate_ind %= 2

                if indicate_ind == 0:
                    gaze_vels = gaze(group, tf_buffer, pids2)
                elif indicate_ind == 1:
                    gaze_vels = gaze(group, tf_buffer, pids2, base_part_world)                    
                
                breathe_vels = breathe(breathe_dict, i, gaze=True)
                i += 1
                i = i%((r/bpm)+1)

                if len(moving_vels) == window_size:
                    moving_vels.pop(0)
                moving_vels.append(breathe_vels)
                vel_cum = sum(moving_vels) / len(moving_vels)
                breathe_vels = vel_cum

                joint_vels = np.concatenate((breathe_vels, gaze_vels))


                if indicate_inner_timer and \
                    (indicate_ind == 0 and time.time() - indicate_inner_timer > ind_inner_time) or \
                    (indicate_ind != 0 and time.time() - indicate_inner_timer > ind_inner_time + time_offset):
                    indicate_inner_timer = None

                vel_msg = Float64MultiArray()
                vel_msg.data = joint_vels
                pub.publish(vel_msg)

                if tasking_flag:
                    indicate_inner_timer = None
                    indicate_ind = 0
                    vel_msg = Float64MultiArray()
                    vel_msg.data = [0., 0., 0., 0., 0., 0.]
                    pub.publish(vel_msg)
                    story_ind += 1    
                    i = 0
                    moving_vels = []
                    pids2 = None
                    tasking_flag = False

            except Exception as e:
                print(e)
                vel_msg = Float64MultiArray()
                vel_msg.data = [0., 0., 0., 0., 0., 0.]
                pub.publish(vel_msg)
                pids2 = None

        elif story_arr[story_ind] == "after_gift":
            # Head movement
            # trajectory = create_traj(last_thanking_pose)
            # client.send_goal(trajectory)
            # result = client.wait_for_result(rospy.Duration(10.0))
            
            
            if not controller == "vel":
                result = switch_controller(['joint_group_vel_controller'], ['pos_joint_traj_controller'], 2, False, 1)
                controller = "vel" if result else "pos"

            if not pids2: pids2 = reset_pids(3)

            try:
                gaze_vels = gaze(group, tf_buffer, pids2)
                breathe_vels = breathe(breathe_dict, i, gaze=True)
                i += 1
                i = i%((r/bpm)+1)

                if len(moving_vels) == window_size:
                    moving_vels.pop(0)
                moving_vels.append(breathe_vels)
                vel_cum = sum(moving_vels) / len(moving_vels)
                breathe_vels = vel_cum

                joint_vels = np.concatenate((breathe_vels, gaze_vels))
                # joint_vels = np.array([breathe_vels[0], breathe_vels[1],breathe_vels[2], gaze_vels[0], gaze_vels[1], 0 ])
                vel_msg = Float64MultiArray()
                vel_msg.data = joint_vels

                pub.publish(vel_msg)
            
                if tasking_flag:
                    vel_msg = Float64MultiArray()
                    vel_msg.data = [0., 0., 0., 0., 0., 0.]
                    pub.publish(vel_msg)
                    story_ind += 1 
                    i = 0
                    moving_vels = []
                    tasking_flag = False
                    pids2 = None

            except Exception as e:
                print(e)
                vel_msg = Float64MultiArray()
                vel_msg.data = [0., 0., 0., 0., 0., 0.]
                pub.publish(vel_msg)

        elif story_arr[story_ind] == "thank2":
            
            vel_msg = Float64MultiArray()
            joint_velocities = np.array([0, 0, 0, nod[nod_i], 0, 0])
            vel_msg.data = joint_velocities
            pub.publish(vel_msg)

            nod_i += 1
            nod_i = nod_i % nod_length
            if nod_i == 0:
                vel_msg.data = np.array([0, 0, 0, 0, 0, 0])
                pub.publish(vel_msg)
                rate.sleep()
                story_ind += 1    

        elif story_arr[story_ind] == "home_last":
            if not controller == "pos":
                result = switch_controller(['pos_joint_traj_controller'], ['joint_group_vel_controller'], 2, False, 1)
                controller = "pos" if result else "vel"
            rospy.sleep(0.1)
            trajectory = create_traj(home_last)
            client.send_goal(trajectory)
            result = client.wait_for_result(rospy.Duration(10.0))
            gripper_action = "close"
            flag_leave_part = "fifth"
            story_ind += 1
            tasking_flag = False
            exit()


        rate.sleep()