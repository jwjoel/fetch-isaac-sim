from omni.isaac.core.controllers import BaseController
import numpy as np
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.core.utils.types import ArticulationAction

class NavigationController(BaseController):
    def __init__(self):
        super().__init__(name="navigation_controller")
        self.target_position = np.array([0.0, 0.0])
        self.target_orientation = 0.0  # target yaw angle
        self.target_set = False
        self.max_speed = 0.4  # maximum linear speed
        self.max_yaw_rate = np.pi / 4  # maximum angular speed
        self.max_acceleration = 0.5  # maximum linear acceleration (m/s^2)
        self.wheel_radius = 0.06  # wWheel radius in meters
        self.wheel_base = 0.32  # distance between wheels in meters
        self.slowing_down_distance = 0.4  # when to start slowing down (meters)
        self.yaw_threshold = np.deg2rad(5)  # yaw error threshold (radians)
        self.yaw_threshold_upper = np.deg2rad(8)
        self.position_threshold = 0.05  # position error threshold (meters)
        self.min_speed = 0.005  # minimum linear speed (m/s)
        self.state = "ROTATE_TO_TARGET_DIRECTION"
        return

    def set_target(self, target_position, target_orientation=None):
        self.target_position = np.array(target_position[:2])
        self.target_orientation = target_orientation
        self.target_set = True
        self.state = "ROTATE_TO_TARGET_DIRECTION"
                
    def forward(self, current_position, current_orientation, current_joint_positions, step_size):
        if not self.target_set:
            # If no target is set, stop the robot by setting wheel velocities to zero
            target_joint_velocities = [0.0, 0.0] + [None] * (len(current_joint_positions) - 2)
            return ArticulationAction(joint_velocities=target_joint_velocities)

        current_position_xy = np.array(current_position[:2])
        direction_vector = self.target_position - current_position_xy
        distance_to_target = np.linalg.norm(direction_vector)

        # Normalize direction vector if distance is significant
        if distance_to_target > 1e-6:
            direction_vector /= distance_to_target
        else:
            # If very close to target, use current orientation to define direction
            direction_vector = np.array([np.cos(self.target_orientation), np.sin(self.target_orientation)])
            distance_to_target = 0.0

        desired_yaw = np.arctan2(direction_vector[1], direction_vector[0])
        current_yaw = quat_to_euler_angles(current_orientation)[2]

        # Normalize yaw difference to [-pi, pi]
        yaw_diff = (desired_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi

        v = 0.0  # Linear speed (m/s)
        omega = 0.0  # Angular speed (rad/s)

        # State machine for controlling the robot's behavior
        if self.state == "ROTATE_TO_TARGET_DIRECTION":
            # Rotate towards the desired yaw
            omega = np.clip(yaw_diff / step_size, -self.max_yaw_rate, self.max_yaw_rate)
            v = 0.0  # No linear motion during rotation

            # Check if the robot is facing the target direction
            if abs(yaw_diff) <= self.yaw_threshold:
                self.state = "MOVE_FORWARD"

        elif self.state == "MOVE_FORWARD":
            # Move forward while correcting yaw towards the target
            yaw_diff = (desired_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
            omega = np.clip(yaw_diff / step_size, -self.max_yaw_rate, self.max_yaw_rate)

            # Adjust linear speed based on distance to target
            if distance_to_target > self.slowing_down_distance:
                v = self.max_speed
            else:
                # Slow down as we approach the target
                v = max(self.min_speed, self.max_speed * (distance_to_target / self.slowing_down_distance))

            # Check if the robot has reached the target position
            if distance_to_target < self.position_threshold:
                self.state = "ROTATE_TO_TARGET_ORIENTATION"

        elif self.state == "ROTATE_TO_TARGET_ORIENTATION":
            # Rotate to the target orientation if specified
            if self.target_orientation is not None:
                target_yaw = self.target_orientation
                yaw_diff = (target_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
                omega = np.clip(yaw_diff / step_size, -self.max_yaw_rate, self.max_yaw_rate)
                v = 0.0  # No linear motion during rotation

                # Check if the robot has reached the target orientation
                if abs(yaw_diff) <= self.yaw_threshold:
                    self.state = "DONE"
            else:
                # No target orientation specified
                self.state = "DONE"

        else:  # "DONE" state
            v = 0.0
            omega = 0.0

        # Convert linear and angular velocities to wheel angular velocities
        v_left = v - (omega * self.wheel_base / 2.0)
        v_right = v + (omega * self.wheel_base / 2.0)
        omega_left = v_left / self.wheel_radius
        omega_right = v_right / self.wheel_radius

        # Prepare the joint velocities array
        target_joint_velocities = [omega_left, omega_right] + [None] * (len(current_joint_positions) - 2)

        return ArticulationAction(joint_velocities=target_joint_velocities)
