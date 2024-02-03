import numpy as np
import copy 

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


class Gaze:
   
    def __init__(self):
       
       # TF's of wrists specific to UR5
        self.H_w1_w2 = np.array([[1, 0, 0, 0],
                                             [0, 0, -1, -0.09465],
                                             [0, 1, 0, 0],
                                             [0, 0, 0, 1]])

        self.H_w2_w3 = np.array([[1, 0, 0, 0],
                                              [0, 0, 1, 0.0823],
                                              [0, -1, 0, 0],
                                              [0, 0, 0, 1]])

    def get_rotation_matrix(self, theta):
      
        """
        Returns a 4x4 rotation matrix based on the given angle.
        """
       
        return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                         [np.sin(theta), np.cos(theta), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def get_distance(self, angles, gaze_point, current_pose, rotation):
        
        """
        Calculates the angle theta based on the desired gaze point, current pose, and rotation.
        """
       
        h_start = self.H_w1_w2 @ self.get_rotation_matrix(angles[1])
        h_end = h_start @ self.H_w2_w3

        h_start = rotation @ self.get_rotation_matrix(angles[0] - current_pose) @ h_start
        h_end = rotation @ self.get_rotation_matrix(angles[0] - current_pose) @ h_end

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

    def get_parallel_distance(self, angle, current_pose, previous_pose, rotation):
       
        """
        Calculates the parallel distance based on the given angle, current pose, previous pose, and transformation matrix.
        """
        
        h_start = self.H_w1_w2 @ self.get_rotation_matrix(current_pose[4])
        h_end = h_start @ self.H_w2_w3 @ self.get_rotation_matrix(angle)

        h_start = rotation @ self.get_rotation_matrix(current_pose[3] - previous_pose) @ h_start
        h_end = rotation @ self.get_rotation_matrix(current_pose[3] - previous_pose) @ h_end

        dot_prod = np.inner(np.array([-1, 0, 0]), h_end[:3, 1])  # Z vector wrt world

        return dot_prod

    def calculate_gaze_poses(self, gaze_point, current_pose, transformation_matrix):
       
        """
        Calculates the gaze poses based on the given gaze point, current pose, and transformation matrix.
        """
       
        initial_angles = np.array([-1.57, -1.57])  # Elbow Down
        res = least_squares(self.get_distance, initial_angles, method='dogbox',
                            bounds=(-1.9 * np.pi, 1.9 * np.pi),
                            args=(np.array([gaze_point]), current_pose[3], transformation_matrix))
        theta_1, theta_2 = res['x']
        previous_pose = current_pose[3]
        current_pose[3] = theta_1
        current_pose[4] = theta_2

        initial_angle = -1.57  # Elbow Down
        res = least_squares(self.get_parallel_distance, initial_angle, method='dogbox',
                            bounds=(-np.pi, np.pi),
                            args=(current_pose, previous_pose, transformation_matrix))
        current_pose[5] = res['x'][0]
        return current_pose

    def calculate_gaze_velocities(self, group, tf_buffer, target=None):
        """
        Calculates the gaze velocities based on the current joint values, transformation data, and target (if provided).
        """
        joint_states = group.get_current_joint_values()
        try:
            transformation = tf_buffer.lookup_transform("world", "wrist_1_link", rospy.Time())
            rotation = R.from_quat(np.array([transformation.transform.rotation.x,
                                             transformation.transform.rotation.y,
                                             transformation.transform.rotation.z,
                                             transformation.transform.rotation.w]))
                                             
            rotation_matrix = rotation.as_matrix()
            rotation_matrix = np.vstack((rotation_matrix, [0, 0, 0]))
            rotation_matrix = np.hstack((rotation_matrix, np.array([[transformation.transform.translation.x,
                                                                      transformation.transform.translation.y,
                                                                      transformation.transform.translation.z,
                                                                      1.0]]).T))

            if target is None:
                gaze_point_pose = tf_buffer.lookup_transform("world", "eye", rospy.Time())
                gaze_point = gaze_point_pose.transform.translation
                gaze_point = np.array([gaze_point.x, gaze_point.y, gaze_point.z, 1])
            else:
                target = np.array(target)
                gaze_point = np.hstack((target, [0]))

            desired_joints = np.array(calculate_gaze_poses(gaze_point, copy.deepcopy(joint_states), rotation_matrix))
        except Exception as e:
            desired_joints = np.array([0, 0, 0, joint_states[3], joint_states[4], joint_states[5]])

        pids[0].setpoint = desired_joints[3]
        pids[1].setpoint = desired_joints[4]
        pids[2].setpoint = desired_joints[5]
        return np.array([pids[0](joint_states[3]), pids[1](joint_states[4]), pids[2](joint_states[5])])