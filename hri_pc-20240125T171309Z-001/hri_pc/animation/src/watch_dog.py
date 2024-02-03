import rospy
import moveit_commander
import os
import math
import actionlib

from controller_manager_msgs.srv import ListControllers
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import FollowJointTrajectoryAction

if __name__=="__main__":
    rospy.init_node("watch_dog")
    group = moveit_commander.MoveGroupCommander("manipulator")
    rospy.wait_for_service("controller_manager/list_controllers")
    traj_client = actionlib.SimpleActionClient("pos_joint_traj_controller/follow_joint_trajectory", 
                                            FollowJointTrajectoryAction)
    client = rospy.ServiceProxy("controller_manager/list_controllers",
                                            ListControllers)
    pub = rospy.Publisher("joint_group_vel_controller/command", Float64MultiArray, queue_size=1)

    print("Connected to controller manager")
    
    rate = rospy.Rate(10)
    vel_controller_state = None
    pos_traj_controller_state = None
    limit_r = 0.7
    while not rospy.is_shutdown():
        controllers = client()
        for controller in controllers.controller:
            if controller.name == "joint_group_vel_controller": vel_controller_state = controller.state
            elif controller.name == "pos_joint_traj_controller": pos_traj_controller_state = controller.state
        
        current_pose = group.get_current_pose()
        x = current_pose.pose.position.x 
        y = current_pose.pose.position.y
        r = math.sqrt(x**2 + y**2)

        if r > limit_r:
            if pos_traj_controller_state == "running":
                traj_client.cancel_all_goals()
            if vel_controller_state == "running":
                vel_msg = Float64MultiArray()
                vel_msg.data = [0., 0., 0., 0., 0., 0.]
                pub.publish(vel_msg)
                nodes = os.popen("rosnode list").readlines()
                for node in nodes:
                    if node == "/experiment\n": 
                        os.system("rosnode kill /experiment")
                        print("killing the experiment node")

        rate.sleep()