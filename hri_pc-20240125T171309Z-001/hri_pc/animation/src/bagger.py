import os
import rospy

from std_msgs.msg import String, Int16MultiArray

def button_callback(data):
    global start_recording
    start_recording = not start_recording
    print("button_callback", start_recording)

def ready_callback(data):
    global exp_id, sess_id
    exp_id, sess_id = data.data

if __name__=="__main__":
    rospy.init_node("logger")
    rospy.Subscriber("button_topic", String, button_callback)
    rospy.Subscriber("ready_topic", Int16MultiArray, ready_callback)

    r = rospy.Rate(10)
    
    start_recording = False
    
    exp_id = -1
    sess_id = -1
    
    while exp_id < 0 and sess_id < 0:
        r.sleep()

    dir = "/home/kovan-robot/animation_exp/{id}".format(id=exp_id)
    if not os.path.isdir(dir): os.mkdir(dir)
    bag_path = os.path.join(dir, "{sess_id}_tfs.bag".format(sess_id=sess_id))
    flag = False
    while not rospy.is_shutdown():
        
        if not flag and start_recording:
            print("started bagging")
            os.system("rosbag record /tf /tf_static -O {bag_path}".format(bag_path=bag_path))
            print("here")
            flag = True
        if flag:
            print("exitting")
            exit()

        r.sleep()