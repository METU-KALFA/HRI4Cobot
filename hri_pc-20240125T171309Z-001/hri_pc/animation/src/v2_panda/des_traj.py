import rospy
import moveit_commander
import tf2_ros
from std_msgs.msg import Float64MultiArray
import numpy as np
from sensor_msgs.msg import JointState
import copy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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

if __name__ == "__main__":
    rospy.init_node("traj_control")
    # group = moveit_commander.MoveGroupCommander("manipulator")

    tfBuffer = tf2_ros.Buffer()
    rospy.Subscriber("joint_states", JointState, js_callback)
    listener = tf2_ros.TransformListener(tfBuffer)
    pub = rospy.Publisher("scaled_vel_joint_traj_controller/command", JointTrajectory, queue_size=10) 
    while pub.get_num_connections() == 0:
        rospy.sleep(0.1)

    rate = rospy.Rate(200)

    traj = JointTrajectory()
    traj.header.stamp = rospy.Time.now()
    traj.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    curr_positions = copy.deepcopy(joint_states_global["pos"])
    curr_positions[1] += 0.1
    temp_point = JointTrajectoryPoint()
    traj.points.append(temp_point)
    temp_point.positions.append(curr_positions.tolist())
    temp_point.time_from_start = rospy.Duration(10)
    pub.publish(traj)
    rospy.spin()
