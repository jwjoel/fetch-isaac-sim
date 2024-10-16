import os
from typing import Optional

from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula.kinematics import LulaKinematicsSolver


class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(self, robot_articulation: Articulation, end_effector_frame_name: Optional[str] = None) -> None:
        assets_root_path = '/home/joel/Development/issac/tabletop_rearrange_simulation/fetch/assets'
        self._kinematics = LulaKinematicsSolver(
            robot_description_path=os.path.join(os.path.dirname(__file__), "./rmpflow/robot_descriptor.yaml"),
            urdf_path=assets_root_path + "fetch/fetch.urdf",
        )
        if end_effector_frame_name is None:
            end_effector_frame_name = "wrist_roll_link"
        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)
        return
