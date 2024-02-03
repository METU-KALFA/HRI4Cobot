import rosbag
from matplotlib import pyplot as plt
import rospy
import csv 

def analyze_gaze(bag):
    topics = set()
    prev_t = rospy.Duration(0)
    counter = 0
    prev_state = 0
    button_counter = 0
    tick = None

    bolt_id = 0
    gaze_in_sess = []
    gaze_in_bolt = []
    time_in_bolt = []

    stroy_line_flag = False
    first_reach = None
    last_reach = None
    total_gaze = 0.0
    for topic, msg, t in bag.read_messages(topics=("head_state", "story_line")):
        if stroy_line_flag and topic == "story_line" and msg.data == "reach-bolt":
            gaze_in_sess.append(gaze_in_bolt)
            gaze_in_bolt = []
            stroy_line_flag = False
            
            if first_reach is None: first_reach = t.to_sec()
            last_reach = t.to_sec()
            time_in_bolt.append((t-tick_bolt).to_sec())

        elif not stroy_line_flag and topic == "story_line" and msg.data == "wait-wrist":
            stroy_line_flag = True
            tick_bolt = t
       
        if prev_state == 3 and msg.data == 1:
            tick = t
        elif prev_state == 1 and msg.data == 0:
            tock = t
            gaze_in_bolt.append([tick.to_sec(), tock.to_sec()])
        elif prev_state == 1 and msg.data == 4:
            button_counter += 1
        prev_state = msg.data
    if len(gaze_in_sess) == 4:
        gaze_in_sess.append(gaze_in_bolt)
        last_reach += 15.0
        time_in_bolt.append(15.0)

    for bolt in gaze_in_sess[1:]:
        for interaction in bolt:
            total_gaze += interaction[1] - interaction[0]
    
    total_time = 0.0
    for time_one_bolt in time_in_bolt[1:]:
        total_time += time_one_bolt

    ratio = 0.0
    if len(gaze_in_sess): ratio = total_gaze / total_time
    
    return total_gaze, total_time, ratio
if __name__=="__main__":
    # bag = rosbag.Bag('/home/kovan-robot/animation_exp/{}/{}.bag'.format(1, 1)) 
    # topics = set()
    # for topic, msg, t in bag.read_messages():
    #     topics.add(topic)
    # print(topics)

    with open("/home/kovan-robot/animation_exp/gaze_interaction_all.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["exp_id", "sess_id", "total_interaction(s)", "mutual_gaze(s)", "ratio"])
        for exp_id in range(91, 96):
            for sess_id in range(1, 3):
                bag = rosbag.Bag('/home/kovan-robot/animation_exp/{}/{}.bag'.format(exp_id, sess_id)) 
                total_gaze, total_time, ratio = analyze_gaze(bag)
                writer.writerow([exp_id, sess_id, total_time, total_gaze, ratio])
                bag.close()
