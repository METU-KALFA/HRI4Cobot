import numpy as np
import copy
import time
from scipy.interpolate import CubicSpline

class Compensator():
    def __init__(self) -> None:
        self.interrupt_index, self.exit_index, self.time_to_recover  = None, None, None
        self.exit_pos, self.target_pos = None, None
        self.exit_vel, self.target_vel = None, None
        self.n_joints = 3
        self.ff = np.repeat([1], self.n_joints)
        self.kp, self.kd = np.repeat([5], self.n_joints), np.repeat([0.1], self.n_joints)
        self.error, self.d_error = np.zeros(self.n_joints), np.zeros(self.n_joints)
        self.error_matrix = np.zeros((self.n_joints,1)) 
        
    def generate_segment(self, joint_positions, joint_vels, rate):
        self.i = 0
        x = [0, self.time_to_recover]
        
        y_arr = [[y_init, y_target] for y_init, y_target in zip(joint_positions, self.target_pos)]
        bc_arr = [[v_init, v_target] for v_init, v_target in zip(joint_vels, self.target_vel)]
        cs_arr = [CubicSpline(x, y, bc_type=((1, bc[0]), (1, bc[1]))) for y, bc in zip(y_arr, bc_arr)]
        xs = np.arange(x[0], x[1], 1/rate)
        self.vals = [cs(xs) for cs in cs_arr]
        self.speeds = [cs(xs, 1) for cs in cs_arr]
        return self.vals, self.speeds

    def reset(self):
        self.i = 0
        self.exit_pos = None
        self.exit_vel = None
        self.error = np.zeros(self.n_joints)
        self.d_error = np.zeros(self.n_joints)

    def compensate(self, joint_positions, joint_vels):
        des_pos = np.array([vals[self.i] for vals in self.vals])
        des_vel = np.array([vals[self.i] for vals in self.speeds])
        error = des_pos - joint_positions
        d_error = des_vel - joint_vels
        self.error_matrix = np.concatenate((self.error_matrix, error.reshape(self.n_joints,1)), axis=1)
        command = des_vel*self.ff + error*self.kp + d_error*self.kd
        self.i += 1
        return command


class Breather:
    
    def __init__(self, breathe_dict) -> None:
        assert "control_rate" in breathe_dict.keys(), "Control rate is not added into the breahte_dict."
        self.control_rate = breathe_dict["control_rate"]

        assert "breathe_vec" in breathe_dict.keys(), "breathe_vec does not exist in breathe dictionary."
        self.breathe_vec = np.concatenate((breathe_dict["breathe_vec"], [0, 0, 0]))
        
        assert "f" in breathe_dict.keys(), "f function for breathing velocity profile does not exist in breathe dictionary."
        self.f = breathe_dict["f"]

        try:
            self.freq = breathe_dict["freq"]
        except:
            self.freq = 4.0
            print(f"freq is not given in the breathe dict. It is set to {self.freq}")

        # Give index for each data point to interpolate in the main loop
        self.num_of_vel_pts = self.f.shape[0]
        self.indices = np.linspace(1, self.num_of_vel_pts, self.num_of_vel_pts)

        try:
            self.amplitude = breathe_dict["amplitude"]
        except:
            self.amplitude = 80.0
            print(f"amplitude is not given in the breathe dict. It is set to {self.amplitude}")


        if "num_of_joints" in breathe_dict.keys():
            self.num_of_joints = breathe_dict["num_of_joints"]
        else:
            self.num_of_joints = 3  # Body joints for UR

        if "compensation" in breathe_dict.keys():
            self.compensation = breathe_dict["compensation"]
        else:
            self.compensation = True

        if self.compensation: 
            self.compensator = Compensator()
            self.compensator.n_joints = self.num_of_joints
            if "interrupt_index" in breathe_dict.keys():
                self.compensator.interrupt_index = int(breathe_dict["interrupt_index"] * int(self.control_rate/self.freq))
            else:
                self.compensator.interrupt_index = int(0.45 * int(self.control_rate/self.freq))
            
            if "exit_index" in breathe_dict.keys():
                self.compensator.exit_index = int(breathe_dict["exit_index"] * int(self.control_rate/self.freq))
            else:
                self.compensator.exit_index = int(0.60 * int(self.control_rate/self.freq))

            assert self.compensator.interrupt_index < self.compensator.exit_index, "Interrupt index should be smaller than exit index."

            self.exit_positions = []
            self.exit_vels = []

        self.loop_index = 0
        self.breathe_count = 0

    def step_breathe(self, joint_positions, jacobian_func):
        # interpolate velocity from f function (human data)
        velocity_magnitude = np.interp(self.num_of_vel_pts * self.freq * self.loop_index / self.control_rate, 
                                        self.indices, 
                                        self.f)
        # Breathe velocity in task space
        velocity_task = self.breathe_vec * velocity_magnitude * self.freq * self.amplitude
        jacobian = jacobian_func(joint_positions.tolist())
        # rcond may be tuned not to get closer to singularities.
        # Take psuedo inverse of Jacobian.
        pinv_jacobian = np.linalg.pinv(jacobian, rcond=1e-15)  
        # Task space --> Configuration space 
        velocity_command = np.dot(pinv_jacobian[:self.num_of_joints], velocity_task)
        
        return velocity_command
    
    def step(self, joint_positions, joint_vels, jacobian_func):
        if self.compensation:
            if self.breathe_count == 0:
                if self.loop_index == self.compensator.exit_index:
                    self.compensator.time_to_recover = (self.compensator.exit_index - self.compensator.interrupt_index) / self.control_rate
                    self.compensator.target_pos = copy.deepcopy(joint_positions[:self.num_of_joints])
                    self.compensator.target_vel = copy.deepcopy(joint_vels[:self.num_of_joints])

            if self.breathe_count > 0 \
                and self.loop_index >= self.compensator.interrupt_index \
                and self.loop_index < self.compensator.exit_index:

                if self.loop_index == self.compensator.interrupt_index:
                    self.compensator.generate_segment(joint_positions[:self.num_of_joints], joint_vels[:self.num_of_joints], self.control_rate)  # Generates cubic spline

                self.compensator.joints_states = {"pos": joint_positions, "vels": joint_vels}
                velocity_command = self.compensator.compensate(joint_positions[:self.num_of_joints], joint_vels[:self.num_of_joints])
            else:
                velocity_command = self.step_breathe(joint_positions, jacobian_func)
        
        else:
            velocity_command = self.step_breathe(joint_positions, jacobian_func)

        self.loop_index = self.loop_index + 1 
        self.loop_index = self.loop_index % int(self.control_rate / self.freq)
        if self.loop_index == 0: 
            self.breathe_count += 1
            print(f"Breathe count: {self.breathe_count}")

        return velocity_command