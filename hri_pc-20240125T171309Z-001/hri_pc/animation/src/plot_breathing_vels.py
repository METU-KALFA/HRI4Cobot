import rospy
import numpy as np
import moveit_commander
import scipy.signal as sig

import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

def js_callback(msg):
    global joint_vels
    joint_vels = list(msg.velocity)
    temp = joint_vels[0]
    joint_vels[0] = joint_vels[2]
    joint_vels[2] = temp

if __name__=="__main__":
    # rospy.init_node("record_ee_vel")
    # group = moveit_commander.MoveGroupCommander("manipulator")
    # rospy.Subscriber("joint_states", JointState, js_callback)
    # publisher = rospy.Publisher("/vel_mag", Float64)
    # r = 30
    # rate = rospy.Rate(r)
    
    # joint_vels = np.array([0., 0., 0., 0., 0., 0.])
    
    # poses = []
    # vels = []
    # start = rospy.get_time()
    # while not rospy.is_shutdown(): # and rospy.get_time() - start < 10:
    #     # ee_pose = group.get_current_pose()
    #     # poses.append([ee_pose.pose.position.x, ee_pose.pose.position.y, ee_pose.pose.position.z])
    #     joint_states = group.get_current_joint_values()
    #     jacobian = group.get_jacobian_matrix(joint_states)  # np array 6x6
    #     vel_cart = np.dot(jacobian, joint_vels)
    #     publisher.publish(np.linalg.norm(vel_cart))
    #     # vels.append(vel_cart[:3])
    #     rate.sleep()

    # # poses = np.array(poses)
    # # np.save("vels.npy", vels)
    # vels = np.load("vels.npy")
    # poses = np.load("poses.npy")
    # dirr = []
    
    shape_data = np.array([0.0, 0, 0.002624999999999991, 0.005624999999999991, 0.014378906250000045, 
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
            
    # shape_vel = np.array([shape_data[i+1]-shape_data[i] for i in range(len(shape_data)-1)])
    shape_vel = np.array([shape_data[i+1]-shape_data[i] for i in range(len(shape_data)-1)])
    shape_vel = np.hstack(([shape_data[-1]-shape_data[0]], shape_vel))

    for i in range(len(shape_vel)):
        if i == 0: 
            continue
        elif i == len(shape_vel) - 1:
            shape_vel[i] = (2*shape_vel[i-1] + shape_vel[i])/3.0 
        else: 
            shape_vel[i] = (2*shape_vel[i-1] + shape_vel[i] + 2*shape_vel[i+1])/5.0

    r = 30
    sec = 8
    amplitude = 30.
    period = 4
    bpm = 1.0/period
    breathe_vec = np.array([0.7, 0, 0.3])
    num_of_vel_pts = shape_vel.shape[0]
    indices = np.linspace(0, num_of_vel_pts-1, num_of_vel_pts)
    given_vels = []
    moving_vels = []
    window_size = 10
    vel_cum = [0., 0., 0., 0., 0., 0.]
    for i in range(r*sec):
        k = i%((r/bpm)+1)
        vel_mag = np.interp(num_of_vel_pts*bpm*k/r, indices, shape_vel)
        vel = breathe_vec * vel_mag * bpm * amplitude
        if len(moving_vels) == window_size:
            moving_vels.pop(0)    
        moving_vels.append(vel)
        vel_cum = sum(moving_vels) / len(moving_vels)
        given_vels.append(np.linalg.norm(vel_cum))

    # vels_orig = shape_vel
    # shape_acc = np.array([shape_vel[i+1]-shape_vel[i] for i in range(len(shape_vel)-1)])
    # # for i in range(len(poses)-1):
    # for i in range(len(vels)):
    #     # vels.append(poses[i+1] - poses[i])
    #     if vels[i][0] < 0: dirr.append(1)
    #     else: dirr.append(-1)
    # # vels = np.array(vels)
    # dirr = np.array(dirr)
    # vels = vels[84:]
    # dirr = dirr[84:]
    
    # vels_mag = dirr * np.linalg.norm(vels, axis=1)
    # poses_mag = np.linalg.norm(poses, axis=1)
    # # np.save("poses.npy", poses)
    fig, ax = plt.subplots()
    # ax.plot(vels_mag)
    ax.plot(shape_vel)
    plt.show()