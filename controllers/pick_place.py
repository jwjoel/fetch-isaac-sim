from tokenize import String
from .pick_place_base import PickPlaceControllerBase
from omni.isaac.core.articulations import Articulation
from omni.isaac.manipulators.grippers import ParallelGripper

from .rmpflow import RMPFlowController
import numpy as np

class PickPlaceController(PickPlaceControllerBase):
    def __init__(self, name: str, gripper: ParallelGripper, robot_articulation: Articulation, robot_name: str, events_dt=None, operation_mode: str = 'pick_place') -> None:
        if events_dt is None:
            events_dt = [0.008, 0.0075, 0.1, 0.1, 0.005, 0.001, 0.0025, 1, 0.008, 0.08]
        super().__init__(
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=0.7,
            operation_mode=operation_mode,
        )
        self.task_parameters_set = False
        self.picking_position = None
        self.placing_position = None
        self.end_effector_orientation = None
        self.initial_end_effector_position = None
        self.robot_name = robot_name

    def set_pick_parameters(self, picking_position, end_effector_orientation):
        self.reset()
        self.picking_position = picking_position
        self.end_effector_orientation = end_effector_orientation
        self.task_parameters_set = True
        self.operation_mode = 'pick'

    def set_place_parameters(self, placing_position, end_effector_orientation, initial_end_effector_position):
        self.reset()
        self.placing_position = placing_position
        self.end_effector_orientation = end_effector_orientation
        self.initial_end_effector_position = initial_end_effector_position
        self.task_parameters_set = True
        self.operation_mode = 'place'

    def set_pick_place_parameters(self, picking_position, placing_position, end_effector_orientation):
        self.reset()
        self.picking_position = picking_position
        self.placing_position = placing_position
        self.end_effector_orientation = end_effector_orientation
        self.task_parameters_set = True
        self.operation_mode = 'pick_place'

    def reset(self):
        super().reset()

    def forward(self, observations):
        if not self.task_parameters_set:
            raise ValueError("Task parameters not set. Call set_task_parameters first.")

        return super().forward(
            picking_position=self.picking_position,
            placing_position=self.placing_position,
            current_joint_positions=observations[self.robot_name]["joint_positions"],
            initial_end_effector_position=self.initial_end_effector_position,
            end_effector_orientation=self.end_effector_orientation,
            end_effector_offset=np.array([0, 0, 0.002]),
        )

    def is_done(self):
        done = super().is_done()
        if done:
            self.task_parameters_set = False
            self.picking_position = None
            self.placing_position = None
            self.end_effector_orientation = None
            self.initial_end_effector_position = None
        return done