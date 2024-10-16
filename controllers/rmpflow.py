import os

import omni.isaac.motion_generation as mg
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.objects import FixedCuboid
import numpy as np

class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0) -> None:
        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=os.path.join(os.path.dirname(__file__), "../rmpflow/robot_descriptor.yaml"),
            rmpflow_config_path=os.path.join(os.path.dirname(__file__), "../rmpflow/fetch_rmpflow_common.yaml"),
            urdf_path=os.path.join(os.path.dirname(__file__), "../assets/fetch/fetch.urdf"),
            end_effector_frame_name="gripper_link",
            maximum_substep_size=0.00334,
        )
                
        self._obstacle = FixedCuboid("/World/obstacle",size=0.93,position=np.array([1.0, 0.0, 0.5316]),color=np.array([0.,0.,1.]), scale=np.array([1.0, 1.0, 0.015]))
        
        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmpflow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        (
            self._default_position,
            self._default_orientation,
        ) = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        
        mg.MotionPolicyController.add_obstacle(self, obstacle=self._obstacle)

        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
