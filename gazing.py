import numpy as np

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
