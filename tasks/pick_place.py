import os
from typing import Optional

import numpy as np
from .pick_place_base import PickPlaceBase
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.nucleus import get_assets_root_path


class PickPlace(PickPlaceBase):
    def __init__(
        self,
        name: str = "fetch_pick_place",
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        PickPlaceBase.__init__(
            self,
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=np.array([0.05, 0.05, 0.05]),
            offset=offset,
        )
        return
    
    def set_robot(self) -> SingleManipulator:
        assets_root_path = '/home/joel/.local/share/ov/pkg/isaac-sim-4.1.0/projects/tabletop_rearrangement/assets'
        asset_path = assets_root_path + "/fetch/fetch_new.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World")
        gripper = ParallelGripper(
            end_effector_prim_path="/World/fetch/gripper_link",
            joint_prim_names=["l_gripper_finger_joint", "r_gripper_finger_joint"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.00, 0.00]),
            action_deltas=np.array([0.01, 0.01]),
        )
        manipulator = SingleManipulator(
            prim_path="/World/fetch",
            name="fetch_robot",
            end_effector_prim_name="gripper_link",
            gripper=gripper,
        )
        joints_default_positions = np.zeros(14)
        joints_default_positions[12] = 0.05
        joints_default_positions[13] = 0.05
        manipulator.set_joints_default_state(positions=joints_default_positions)
        return manipulator

    # def set_robot(self) -> SingleManipulator:
    #     assets_root_path = '/home/joel/.local/share/ov/pkg/isaac-sim-4.1.0/projects/tabletop_rearrangement/assets'
    #     asset_path = assets_root_path + "/fetch/fetch_new.usd"
    #     add_reference_to_stage(usd_path=asset_path, prim_path="/World")
    #     gripper = ParallelGripper(
    #         end_effector_prim_path="/World/fetch/wrist_roll_link",
    #         joint_prim_names=["l_gripper_finger_joint", "r_gripper_finger_joint"],
    #         joint_opened_positions=np.array([0.05, 0.05]),
    #         joint_closed_positions=np.array([0.00, 0.00]),
    #         action_deltas=np.array([0.01, 0.01]),
    #     )
    #     manipulator = SingleManipulator(
    #         prim_path="/World/fetch",
    #         name="fetch_robot",
    #         end_effector_prim_name="wrist_roll_link",
    #         gripper=gripper,
    #     )
    #     joints_default_positions = np.zeros(14)
    #     joints_default_positions[12] = 0.05
    #     joints_default_positions[13] = 0.05
    #     manipulator.set_joints_default_state(positions=joints_default_positions)
    #     return manipulator
