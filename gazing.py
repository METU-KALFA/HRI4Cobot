""" Gazing code for UR5 robot. 
It minimizes the angle between two vectors, namely current gaze vector and desired gaze vector.

@project TUBITAK Kalfa  
@author Burak Bolat
@email burakbolatcs@gmail.com
@copyright '2022, Burak Bolat'
"""

import numpy as np
from simple_pid import PID
from scipy.optimize import least_squares

def reset_pids(Kp=1.0, Ki=0.0, Kd=0.0):
        # Kp, Ki, Kd = (5.0, 0.1, 0.1)
        # Kp, Ki, Kd = (3.0, 0.05, 0.0)
        Ki, Kd = (0.05, 0.0)
        pid_w1 = PID(Kp, Ki, Kd, setpoint=0.0)
        pid_w2 = PID(Kp, Ki, Kd, setpoint=0.0)
        pid_w3 = PID(Kp, Ki, Kd, setpoint=0.0)
        return [pid_w1, pid_w2, pid_w3]
    
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


class Gazer:
    
    def __init__(self, gaze_dict) -> None:
        if "kp" in gaze_dict.keys():
            self.kp = gaze_dict["kp"]
        else:
            self.kp = 3.0

        if "kd" in gaze_dict.keys():
            self.kd = gaze_dict["kd"]
        else:
            self.kd = 3.0

        if "ki" in gaze_dict.keys():
            self.ki = gaze_dict["ki"]
        else:
            self.ki = 0.0

        assert "initial_guesses" in gaze_dict.keys(), "Initial guesses are not provided. Give them in list format. e.g. [-1.57, -1.57, -3.14]"
        self.initial_guesses = gaze_dict["initial_guesses"]
        

        self.pids = reset_pids(self.kp, self.ki, self.kd)

    def distance(self, x, gaze_point, curr_pose, r):
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

    def distance_parallel(self, x, curr_pose, prev_pose, r):
        h_start = H_w1_w2 @ get_rotation(curr_pose[1])
        h_end = h_start @ H_w2_w3 @ get_rotation(x[0])
        
        h_start = r @ get_rotation(curr_pose[0]-prev_pose) @ h_start
        h_end = r @ get_rotation(curr_pose[0]-prev_pose) @ h_end

        dot_prod = np.inner(np.array([0,0,1]), h_end[:3, 0]) # Z vector wrt world
        
        cos_theta = dot_prod
        theta = np.abs(np.pi/2 - np.arccos(cos_theta))
        return theta

    def desired_joint_angles(self, target, head_tranformation_matrix, joint_states):
        try:
            initial_guess = self.initial_guesses[:2]
            if len(target) == 3: target.append(1.0)
            result = least_squares(self.distance, initial_guess, method='dogbox', bounds=(-1.9*np.pi, 1.9*np.pi), args=(np.array([target]), joint_states[3], head_tranformation_matrix))
            q1, q2 = result['x']
            initial_guess = self.initial_guesses[2]
            result = least_squares(self.distance_parallel, initial_guess, method='dogbox', bounds=(-1.5*np.pi, 1.5*np.pi), args=([q1, q2], joint_states[3], head_tranformation_matrix))
            q3 = result['x'][0]
        except:
            q1, q2, q3 = joint_states[3:]
        return [q1, q2, q3]
    
    def step(self, target, head_tranformation_matrix, joint_states):
        desired_joints = self.desired_joint_angles(target, head_tranformation_matrix, joint_states)
        for pid in self.pids: pid.setpoint = desired_joints.pop(0)
        return [pid(joint_states[i+3]) for i, pid in enumerate(self.pids)]