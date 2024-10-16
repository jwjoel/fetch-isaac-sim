from omni.isaac.core.controllers import BaseController
import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

class NavigationController(BaseController):
    def __init__(self):
        super().__init__(name="navigation_controller")
        self.target_position = np.array([0.0, 0.0])
        self.target_orientation = 0.0  # target yaw angle
        self.target_set = False
        self.max_speed = 1.0  # maximum speed
        self.max_yaw_rate = np.pi / 1
        self.max_acceleration = 0.5  # maximum acceleration
        self.current_speed = 0.0  # current speed
        self.slowing_down_distance = 0.1  # when to slow down
        self.yaw_threshold = np.deg2rad(5)  # yaw error before moving forward
        self.position_threshold = 0.05  # acceptable distance to target before considering arrived
        self.min_speed = 0.1
        self.state = "ROTATE_TO_TARGET_DIRECTION"
        return

    def set_target(self, target_position, target_orientation=None):
        self.target_position = np.array(target_position[:2])
        self.target_orientation = target_orientation
        self.target_set = True
        self.state = "ROTATE_TO_TARGET_DIRECTION"
        
    def forward(self, current_position, current_orientation, step_size):
        if not self.target_set:
            return current_position, current_orientation
        
        current_position_xy = np.array(current_position[:2])
        direction_vector = self.target_position - current_position_xy
        distance_to_target = np.linalg.norm(direction_vector)

        # Normalize direction vector if distance is significant
        if distance_to_target > 1e-6:
            direction_vector /= distance_to_target
        else:
            direction_vector = np.array([np.cos(self.target_orientation), np.sin(self.target_orientation)])
            distance_to_target = 0.0

        desired_yaw = np.arctan2(direction_vector[1], direction_vector[0])
        current_yaw = quat_to_euler_angles(current_orientation)[2]

        # Normalize yaw difference to [-pi, pi]
        yaw_diff = (desired_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi

        new_yaw = current_yaw
        new_position = np.copy(current_position)

        if self.state == "ROTATE_TO_TARGET_DIRECTION":
            # Rotate towards the desired yaw
            yaw_step = np.clip(yaw_diff, -self.max_yaw_rate * step_size, self.max_yaw_rate * step_size)
            new_yaw += yaw_step

            # Check if the robot is facing the target direction
            if abs(yaw_diff) <= self.yaw_threshold:
                self.state = "MOVE_FORWARD"

            # Update orientation
            new_orientation = euler_angles_to_quat([0, 0, new_yaw])

        elif self.state == "MOVE_FORWARD":
            # Move forward and correct yaw towards the target
            # Recalculate the desired yaw based on the updated position
            direction_vector = self.target_position - new_position[:2]
            distance_to_target = np.linalg.norm(direction_vector)
            if distance_to_target > 1e-6:
                direction_vector /= distance_to_target
            desired_yaw = np.arctan2(direction_vector[1], direction_vector[0])
            current_yaw = quat_to_euler_angles(current_orientation)[2]
            yaw_diff = (desired_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi

            # Adjust yaw towards the desired yaw
            yaw_step = np.clip(yaw_diff, -self.max_yaw_rate * step_size, self.max_yaw_rate * step_size)
            new_yaw = current_yaw + yaw_step

            # Update the orientation
            new_orientation = euler_angles_to_quat([0, 0, new_yaw])

            # Adjust speed based on distance to target
            if distance_to_target > self.slowing_down_distance:
                self.current_speed += self.max_acceleration * step_size
                self.current_speed = min(self.current_speed, self.max_speed)
            else:
                self.current_speed -= self.max_acceleration * step_size
                self.current_speed = max(self.current_speed, self.min_speed)

            # Move forward along the new yaw
            move_distance = self.current_speed * step_size
            dx = move_distance * np.cos(new_yaw)
            dy = move_distance * np.sin(new_yaw)
            new_position[0] += dx
            new_position[1] += dy

            # Check if the robot has reached the target position
            new_distance_to_target = np.linalg.norm(self.target_position - new_position[:2])
            if new_distance_to_target < self.position_threshold:
                self.state = "ROTATE_TO_TARGET_ORIENTATION"
                self.current_speed = 0.0  # Stop moving forward

        elif self.state == "ROTATE_TO_TARGET_ORIENTATION":
            # Rotate to the target orientation
            target_yaw = self.target_orientation
            yaw_diff = (target_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
            yaw_step = np.clip(yaw_diff, -self.max_yaw_rate * step_size, self.max_yaw_rate * step_size)
            new_yaw += yaw_step

            # Update orientation
            new_orientation = euler_angles_to_quat([0, 0, new_yaw])

            # Check if the robot has reached the target orientation
            if (target_yaw == None):
                self.state = "DONE"
            elif abs(yaw_diff) <= self.yaw_threshold:
                self.state = "DONE"
        else:  # "DONE" state
            # Robot has reached its destination and orientation
            new_orientation = current_orientation

        return new_position, new_orientation
