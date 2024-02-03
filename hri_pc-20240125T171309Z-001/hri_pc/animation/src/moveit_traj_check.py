# import roslib
import rospy
import actionlib
import moveit_commander
import time
import tf2_ros
import numpy as np
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import copy
from scipy.spatial.transform import Rotation as R


def connect_service(name):
    client = actionlib.SimpleActionClient(name,
                                            FollowJointTrajectoryAction)
    print(client.wait_for_server(timeout=rospy.Duration(1.0)))
    print("Connected to trajectory manager")
    return client

def main():
    group = moveit_commander.MoveGroupCommander("manipulator")
    rospy.init_node("check_inv_jacobian")
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    client = connect_service("vel_based_pos_traj_controller/follow_joint_trajectory")
    rospy.sleep(1)
    
    while not rospy.is_shutdown():
        a = input("num: ")
        ##########
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.points.append(JointTrajectoryPoint())
        goal.trajectory.points.append(JointTrajectoryPoint())
        goal.trajectory.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        
        current = group.get_current_joint_values()
        print(current)
        goal.trajectory.points[0].positions = current
        goal.trajectory.points[1].positions = [1.57, 0.9, -2.1, -1.57, -1.57, -1.57]
        
        goal.trajectory.header.stamp = rospy.Time.now()# + rospy.Duration(1)
        goal.trajectory.points[0].time_from_start = rospy.Duration(0.05)
        goal.trajectory.points[1].time_from_start = rospy.Duration(float(a))
        
        client.send_goal(goal)
        print("the goal is sent")
        # Wait for up to 5 seconds for the motion to complete
        result = client.wait_for_result(rospy.Duration(100.0))
        ###########

        start = group.get_current_pose().pose
        trans = tfBuffer.lookup_transform('base_link', 'ee_link', rospy.Time(0))
        quat = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
        roti = R.from_quat(quat)
        roti = np.dot(roti.as_dcm(), [0,1,0])
        breathe_vec = np.concatenate((roti, [0,0,0]))
        
        shape_data = [0.0, 0.005624999999999991, 0.014378906250000045, 
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
            0.0006273850763798272, 0.00044368593276999935]
        max_data = max(shape_data)
        waypoints = []
        for data in shape_data:
            pose_next = copy.deepcopy(start)
            pose_next.position.x = start.position.x \
                + data*breathe_vec[0]/max_data*(1./15)
            pose_next.position.y = start.position.y \
                + data*breathe_vec[1]/max_data*(1./15)
            pose_next.position.z = start.position.z \
                + data*breathe_vec[2]/max_data*(1./15)
            waypoints.append(copy.deepcopy(pose_next))
            # print(data*breathe_vec[0]/max_data)
            # print(data*breathe_vec[1]/max_data)
            # print(data*breathe_vec[2]/max_data)
            # print("#############")
        trajectory, fraction = group.compute_cartesian_path(waypoints, eef_step=0.01, jump_threshold=0)
        print(float(trajectory.joint_trajectory.points[0].time_from_start.nsecs)/1e9)
        print(trajectory.joint_trajectory.points[-1].time_from_start.secs, float(trajectory.joint_trajectory.points[-1].time_from_start.nsecs)/1e9)
        print(trajectory.joint_trajectory.points[-1].time_from_start - trajectory.joint_trajectory.points[0].time_from_start)
        print("##################")
        trajectory.joint_trajectory.points.remove(trajectory.joint_trajectory.points[0])
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = copy.deepcopy(trajectory.joint_trajectory)
        client.send_goal(goal)
        result = client.wait_for_result(rospy.Duration(100.0))

if __name__=="__main__": main()