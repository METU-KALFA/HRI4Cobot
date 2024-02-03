""" Gazing code for UR5 robot. 
It minimizes the angle between two vectors, namely current gaze vector and desired gaze vector.

@project TUBITAK Kalfa  
@author Burak Bolat
@copyright '2022, Burak Bolat'
"""

# Robot related libraries
import rospy
import moveit_commander

import tf2_ros
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String
from visualization_msgs.msg import InteractiveMarkerFeedback
import utils

# Gaze related libraries
import numpy as np
from scipy.optimize import least_squares

import time
import cv2
import datetime
import os

last_callback_time = 0
img_counter = 1

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
pose_file = f"images/poses_{current_time}.txt"
os.makedirs("images", exist_ok=True)

cam = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

H_w1_w2 = np.array([[1, 0, 0, 0],
                    [0, 0, -1, -0.09465],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]])

H_w2_w3 = np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0.0823],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]])

def get_rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                    [np.sin(theta), np.cos(theta), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

def dist(x, gaze_point, curr_pose, r):
    h_start = H_w1_w2 @ get_rotation(x[1])
    h_end = h_start @ H_w2_w3  # Note that wrong rotation. Rotation of the last joint (wrist_3) omitted.
    
    h_start = r @ get_rotation(x[0]-curr_pose) @ h_start
    h_end = r @ get_rotation(x[0]-curr_pose) @ h_end

    start = h_start[:3, -1]
    end = h_end[:3, -1]
    current_dir = end - start
    current_dir /= np.linalg.norm(current_dir)

    gaze_point = gaze_point.T[:3].ravel()
    desired_dir = gaze_point - start
    desired_dir /= np.linalg.norm(desired_dir)

    dot_prod = np.inner(desired_dir, current_dir)
    cos_theta = dot_prod
    theta = np.arccos(cos_theta)
    return theta

def parallel_dist(x, curr_pose, prev_pose, r, show=False):
    h_start = H_w1_w2 @ get_rotation(curr_pose[4])
    h_end = h_start @ H_w2_w3 @ get_rotation(x[0])
    
    h_start = r @ get_rotation(curr_pose[3]-prev_pose) @ h_start
    h_end = r @ get_rotation(curr_pose[3]-prev_pose) @ h_end

    dot_prod = np.inner(np.array([0,0,1]), h_end[:3, 1])
    
    if show:
        print(h_end)
        print("--------++++++++++-w---------")
    
    return dot_prod

def gui_subs(data, args):
    if data.data == "Gaze": gaze(args)

def gaze(target_position):
    tick = time.time()

    transformation = tf_buffer.lookup_transform("base_link", "wrist_1_link", rospy.Time())
    r = R.from_quat(np.array([transformation.transform.rotation.x,
                            transformation.transform.rotation.y,
                            transformation.transform.rotation.z,
                            transformation.transform.rotation.w]))
    r = r.as_matrix()
    r = np.vstack((r, [0,0,0]))
    r = np.hstack((r, np.array([[transformation.transform.translation.x,
                            transformation.transform.translation.y,
                            transformation.transform.translation.z,
                            1.0]]).T))
    # gaze_point_pose = tf_buffer.lookup_transform("world", "human_wrist/base_link", rospy.Time())
    # gaze_point = gaze_point_pose.transform.translation 
    # gaze_point = np.array([gaze_point.x, gaze_point.y, gaze_point.z, 1])

    gaze_point = np.array(target_position + [1])


    # gaze_point = np.hstack((drawer.end, np.array([1])))
    curr_pose = group.get_current_joint_values()
    
    x0 = np.array([-1.57, -1.57])  # Elbow Down
    x0 = np.array([-4.0, -1.57])  # Elbow Up 1
    res = least_squares(dist, x0, method='dogbox', bounds=(-1.9*np.pi, 1.9*np.pi), args=(np.array([gaze_point]), curr_pose[3], r))
    theta_1, theta_2 = res['x']
    prev_pose = curr_pose[3]
    curr_pose[3] = theta_1
    curr_pose[4] = theta_2
    x0 = -1.57
    res = least_squares(parallel_dist, x0, method='dogbox', bounds=(-np.pi, np.pi), args=(curr_pose, prev_pose, r))
    curr_pose[5] = res['x'][0]
    # print(res)
    return curr_pose 
    # parallel_dist([curr_pose[5]], curr_pose, prev_pose, r, True)

def gaze_vel(gaze_point, curr_pose, r):
    
    # x0 = np.array([-1.57, -1.57])  # Elbow Down
    x0 = np.array([-4.0, -1.57])  # Elbow Up 1
    # x0 = np.array([-1.27, -4.65])  # Elbow Up 2
    res = least_squares(dist, x0, method='dogbox', bounds=(-1.9*np.pi, 1.9*np.pi), args=(np.array([gaze_point]), curr_pose[3], r))
    theta_1, theta_2 = res['x']
    prev_pose = curr_pose[3]
    curr_pose[3] = theta_1
    curr_pose[4] = theta_2
    x0 = -1.57  # Elbow Down and Up 1
    # x0 = 1.57  # Elbow Up 2
    res = least_squares(parallel_dist, x0, method='dogbox', bounds=(-np.pi, np.pi), args=(curr_pose, prev_pose, r))
    # print(res)
    curr_pose[5] = res['x'][0]
    return curr_pose
    

def callback(data):
    global last_callback_time
    current_time = time.time()
    if current_time - last_callback_time > 0.5:  # 500 ms
        last_callback_time = current_time
        position = data.pose.position
        x = 0.387
        y = -0.386
        z = -0.305
        gaze([position.x, position.y, position.z])
    
def save_state():
    global pose_file, img_counter, cam
    transformation = tf_buffer.lookup_transform("base_link", "tool0_controller", rospy.Time())
    # print(group.get_current_pose())
    # print(transformation.transform.translation)
    rospy.sleep(0.2)
    # if cam is None or not cam.isOpened():
    #     print("Failed to open cam")
    res, frame = cam.read()
    if not res:
        print("Failed")
    else:
        # cv2.imshow("test", frame)
        cv2.namedWindow("Press ESC to save image and close window")
        rospy.sleep(5)
        # Press ESC to escape after the cam is ready.
        while True:
            res, frame = cam.read()
            cv2.imshow("Press ESC to save image and close window", frame)
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC Pressed
                # print("Camera has been initialized.")
                cv2.destroyAllWindows()
                break
        img_name = "images/image{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        with open(pose_file, "a") as poses:
            poses.write(f"{img_name} Poses:\n")
            poses.write(str(transformation.transform) + "\n")
        print("{} written!".format(img_name))
        img_counter += 1

if __name__=="__main__":
    rospy.init_node('gazing')
    group = moveit_commander.MoveGroupCommander("manipulator")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    rospy.sleep(0.1)

    # Initial Position values for given object (where to gaze)
    x = 0.387
    y = -0.386
    z = -0.305

    # Camera Initialization
    # Camera shows green screen until frame is captured 10 times. Do not know why??
    # for i in range(10):
    #     res, frame = cam.read()
    #     img_name = "images/image0.png"
    #     cv2.imwrite(img_name, frame)
    # cv2.namedWindow("test")
    # cv2.imshow("test", frame)
    # cv2.waitKey(5)
    # cv2.destroyAllWindows()
    # print("Camera has been initialized.")

    # cv2.namedWindow("test")
    # rospy.sleep(5)
    # # Press ESC to escape after the cam is ready.
    # while True:
    #     cv2.imshow("test", frame)
    #     k = cv2.waitKey(1)
    #     if k%256 == 27:
    #         # ESC Pressed
    #         print("Camera has been initialized.")
    #         cv2.destroyAllWindows()
    #         break

    input("Continue by pressing enter")
    # pose_target = gaze([x, y, z])
    # group.go(pose_target)

    # TO-DO interpolation of joint states
    # 1st ellipse
    midpoint = np.array([5.2455902099609375, -1.3558066350272675, 1.2684066931353968])
    midwhite = np.array([4.450909614562988, -0.16073031843219, 0.6908558050738733])
    midblack = np.array([6.208501815795898, -0.2544592183879395, 0.8468311468707483])
    centertop = np.array([5.294220447540283, -0.9982021015933533, 0.49705249467958623])
    cornerw1 = np.array([4.081981658935547, -0.7202744048884888, 1.914727036152975])
    cornerw2 = np.array([4.952577590942383, 0.019649310702941847, 0.35308581987489873])
    cornerb1 = np.array([6.193326950073242, -0.8232862514308472, 2.0031564871417444])
    cornerb2 = np.array([5.671020030975342, -0.2614482206157227, 0.5529864470111292])
    edgeneut1 = np.array([5.129640579223633, -1.1252576273730774, 2.809300247822897])
    edgeneut2 = np.array([5.444364070892334, -0.07446618497882085, 0.2244184652911585])
    sampling_number = 15

    difference = midwhite - midpoint
    current_point = np.copy(midwhite)

    next_pose = group.get_current_joint_values()
    next_pose[0] = current_point[0]
    next_pose[1] = current_point[1]
    next_pose[2] = current_point[2]
    group.go(next_pose)
    input("Continue by pressing enter")

    try:
        pose_target = gaze([x, y, z])
        group.go(pose_target)
    except:
        print("Gaze calculation had problems")

    save_state()
    print(group.get_current_joint_values())
    current_point -= difference/sampling_number
    input("Continue by pressing enter")

    # for i in range(sampling_number):
    #     next_pose = group.get_current_joint_values()
    #     next_pose[0] = current_point[0]
    #     next_pose[1] = current_point[1]
    #     next_pose[2] = current_point[2]
    #     group.go(next_pose)
    #     input("Continue by pressing enter")

    #     try:
    #         pose_target = gaze([x, y, z])
    #         group.go(pose_target)
    #     except:
    #         print("Gaze calculation had problems")

    #     save_state()
    #     print(group.get_current_joint_values())
    #     current_point -= difference/sampling_number
    #     input("Continue by pressing enter")

    # difference = midpoint - midblack
    # current_point = np.copy(midpoint)
    # for i in range(sampling_number):
    #     next_pose = group.get_current_joint_values()
    #     next_pose[0] = current_point[0]
    #     next_pose[1] = current_point[1]
    #     next_pose[2] = current_point[2]
    #     group.go(next_pose)
    #     input("Continue by pressing enter")

    #     try:
    #         pose_target = gaze([x, y, z])
    #         group.go(pose_target)
    #     except:
    #         print("Gaze calculation had problems")

    #     save_state()
    #     print(group.get_current_joint_values())
    #     current_point -= difference/sampling_number
    #     input("Continue by pressing enter")

    # # 2nd ellipse
    # difference = cornerb1 - centertop
    # current_point = np.copy(cornerb1)
    # for i in range(sampling_number):
    #     next_pose = group.get_current_joint_values()
    #     next_pose[0] = current_point[0]
    #     next_pose[1] = current_point[1]
    #     next_pose[2] = current_point[2]
    #     group.go(next_pose)
    #     input("Continue by pressing enter")

    #     try:
    #         pose_target = gaze([x, y, z])
    #         group.go(pose_target)
    #     except:
    #         print("Gaze calculation had problems")

    #     save_state()
    #     print(group.get_current_joint_values())
    #     current_point -= difference/sampling_number
    #     input("Continue by pressing enter")

    # difference = centertop - cornerw2
    # current_point = np.copy(centertop)
    # for i in range(sampling_number):
    #     next_pose = group.get_current_joint_values()
    #     next_pose[0] = current_point[0]
    #     next_pose[1] = current_point[1]
    #     next_pose[2] = current_point[2]
    #     group.go(next_pose)
    #     input("Continue by pressing enter")

    #     try:
    #         pose_target = gaze([x, y, z])
    #         group.go(pose_target)
    #     except:
    #         print("Gaze calculation had problems")

    #     save_state()
    #     print(group.get_current_joint_values())
    #     current_point -= difference/sampling_number
    #     input("Continue by pressing enter")

    # # 3rd ellipse
    # difference = cornerw1 - centertop
    # current_point = np.copy(cornerw1)
    # for i in range(sampling_number):
    #     next_pose = group.get_current_joint_values()
    #     next_pose[0] = current_point[0]
    #     next_pose[1] = current_point[1]
    #     next_pose[2] = current_point[2]
    #     group.go(next_pose)
    #     input("Continue by pressing enter")

    #     try:
    #         pose_target = gaze([x, y, z])
    #         group.go(pose_target)
    #     except:
    #         print("Gaze calculation had problems")

    #     save_state()
    #     print(group.get_current_joint_values())
    #     current_point -= difference/sampling_number
    #     input("Continue by pressing enter")

    # difference = centertop - cornerb2
    # current_point = np.copy(centertop)
    # for i in range(sampling_number):
    #     next_pose = group.get_current_joint_values()
    #     next_pose[0] = current_point[0]
    #     next_pose[1] = current_point[1]
    #     next_pose[2] = current_point[2]
    #     group.go(next_pose)
    #     input("Continue by pressing enter")

    #     try:
    #         pose_target = gaze([x, y, z])
    #         group.go(pose_target)
    #     except:
    #         print("Gaze calculation had problems")

    #     save_state()
    #     print(group.get_current_joint_values())
    #     current_point -= difference/sampling_number
    #     input("Continue by pressing enter")

    # # 4th ellipse
    # difference = edgeneut2 - centertop
    # current_point = np.copy(edgeneut2)
    # for i in range(sampling_number):
    #     next_pose = group.get_current_joint_values()
    #     # print(next_pose)
    #     next_pose[0] = current_point[0]
    #     next_pose[1] = current_point[1]
    #     next_pose[2] = current_point[2]
    #     group.go(next_pose)
    #     input("Continue by pressing enter")

    #     try:
    #         pose_target = gaze([x, y, z])
    #         group.go(pose_target)
    #     except:
    #         print("Gaze calculation had problems")

    #     save_state()
    #     print(group.get_current_joint_values())
    #     current_point -= difference/sampling_number
    #     input("Continue by pressing enter")

    # difference = centertop - edgeneut1
    # current_point = np.copy(centertop)
    # for i in range(sampling_number):
    #     next_pose = group.get_current_joint_values()
    #     next_pose[0] = current_point[0]
    #     next_pose[1] = current_point[1]
    #     next_pose[2] = current_point[2]
    #     group.go(next_pose)
    #     input("Continue by pressing enter")

    #     try:
    #         pose_target = gaze([x, y, z])
    #         group.go(pose_target)
    #     except:
    #         print("Gaze calculation had problems")

    #     save_state()
    #     print(group.get_current_joint_values())
    #     current_point -= difference/sampling_number
    #     input("Continue by pressing enter")
    
    input("Continue to get current joint values pressing enter")

    print(group.get_current_joint_values())
    
    input("Quit by pressing enter")
    # cv2.destroyAllWindows()
    cam.release()
    exit()
    rate = rospy.Rate(120)  # Set the rate to 60 Hz
    # while not rospy.is_shutdown():
    #     rate.sleep()