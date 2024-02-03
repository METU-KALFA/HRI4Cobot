import numpy as np
import matplotlib.pyplot as plt 


import rosbag

from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt

bag_filename = "/home/kovan3/ur5_ws/src/Exps/exp003/joint_states_bag"  
topic_name = "joint_states"      

positions = {}
velocities = {}

with rosbag.Bag(bag_filename, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if topic == topic_name:
            for i in range(len(msg.name)):
                joint_name = msg.name[i]
                if joint_name not in positions:
                    positions[joint_name] = []
                    velocities[joint_name] = []
                positions[joint_name].append(msg.position[i])
                velocities[joint_name].append(msg.velocity[i])

# Plotting

for joint_name, position_data in positions.items():
    if joint_name == "elbow_joint":
        plt.figure()
        plt.title(f"{joint_name} Position")
        plt.plot(position_data)
        plt.xlabel("Time")
        plt.ylabel("Position (rad)")

for joint_name, velocity_data in velocities.items():
    if joint_name == "elbow_joint":
        plt.figure()
        plt.title(f"{joint_name} Velocity")
        plt.plot(velocity_data)
        plt.xlabel("Time")
        plt.ylabel("Velocity (rad/s)")

plt.show()
# # wrist_pos = np.load("/home/kovan3/burak_breathe/joint_positions_noncomp.npy")
# # fig, axs = plt.subplots(6)
# # axs[0].plot(wrist_pos[:,0])
# # axs[1].plot(wrist_pos[:,1])
# # axs[2].plot(wrist_pos[:,2])
# # axs[3].plot(wrist_pos[:,3])
# # axs[4].plot(wrist_pos[:,4])
# # axs[5].plot(wrist_pos[:,5])
# # plt.show()

# exit_poses = np.load("/home/kovan3/ur5_ws/src/Exps/exp002/exit_positions.npy")
# plt.plot(exit_poses[:,0])
# # plt.plot(exit_poses[:,1])  
# # plt.plot(exit_poses[:,2])
# # plt.plot(exit_poses[:,3])
# # plt.plot(exit_poses[:,4])
# # plt.plot(exit_poses[:,5])
# plt.show()