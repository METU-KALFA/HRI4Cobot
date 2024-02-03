#!/usr/bin/env python

import robotiq_gripper
import rospy
from std_msgs.msg import String

class GripperController:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.gripper = None
        self.init_gripper()
        self.init_ros_node()

    def init_ros_node(self):
        pub = rospy.Publisher('gripper_control', String, queue_size=1)
        rospy.Subscriber("gripper_control", String, self.control_callback)
        rospy.init_node("gripper_state_control", anonymous=True)

    def init_gripper(self):
        print("Creating gripper...")
        self.gripper = robotiq_gripper.RobotiqGripper()
        print("Connecting to gripper...")
        self.gripper.connect(self.host, self.port)
        print("Activating gripper...")
        self.gripper.activate(auto_calibrate=False)

    def control_callback(self, data):
        control_info = data.data
        if "." in control_info:
            try:
                print(control_info)
                control_info = float(control_info)
                control_info = round(255 * (control_info / 0.7))
            except Exception as e:
                self.handle_error()
                return
        else:
            try:
                control_info = int(control_info)
            except Exception as e:
                self.handle_error()
                return

        if control_info > 255:
            control_info = 255
        elif control_info < 0:
            control_info = 0

        try:
            self.gripper.move_and_wait_for_pos(control_info, 255, 255)
        except Exception as e:
            print(e)

    def handle_error(self):
        print("Not supported!")
        print("Use joint state [0 - 0.7] or gripper position [0 - 255]")

if __name__ == "__main__":
    HOST = "10.0.0.2"
    PORT = 63352
    gripper_controller = GripperController(HOST, PORT)
    print("Ready to receive commands.")
    rospy.spin()
