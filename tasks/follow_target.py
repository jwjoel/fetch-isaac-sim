import os
from typing import Optional

import numpy as np
import omni.isaac.core.tasks as tasks
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.core.articulations import Articulation, ArticulationSubset

class FollowTarget(tasks.FollowTarget):
    def __init__(
        self,
        name: str = "fetch_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:
        assets_root_path = '/home/joel/Development/issac/tabletop_rearrange_simulation/fetch/assets'
        asset_path = assets_root_path + "/fetch/fetch.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/fetch")
        gripper = ParallelGripper(
            end_effector_prim_path="/World/fetch/wrist_roll_link",
            joint_prim_names=["l_gripper_finger_joint", "r_gripper_finger_joint"],
            joint_opened_positions=np.array([0.05, 0.05]),
            joint_closed_positions=np.array([0.00, 0.00]),
            action_deltas=np.array([0.05, 0.05]),
        )
        manipulator = SingleManipulator(
            prim_path="/World/fetch",
            name="fetch_robot",
            end_effector_prim_name="wrist_roll_link",
            gripper=gripper,
        )
        joints_default_positions = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        joints_default_positions[12] = 0.05
        joints_default_positions[13] = 0.05
        manipulator.set_joints_default_state(positions=joints_default_positions)

        return manipulator
