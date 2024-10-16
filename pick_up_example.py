# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse

import numpy as np
from controllers.pick_place import PickPlaceController
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from tasks.pick_place import PickPlace
from pxr import PhysxSchema, UsdPhysics
import omni

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

my_world = World(stage_units_in_meters=1.0)

cube_initial_position=np.array([0.7, 0, 0])
target_position = np.array([1.0, 0, 0.5])
# cube_initial_orientation = np.array([0, 0.96006495, -0.01824878, -0.7])
cube_initial_orientation = np.array([0.92388, 0.0, 0.38268, 0.0])
my_task = PickPlace(name="fetch_pick_place", cube_initial_position=cube_initial_position, cube_initial_orientation=cube_initial_orientation, target_position=target_position)

my_world.add_task(my_task)
my_world.reset()
my_fetch = my_world.scene.get_object("fetch_robot")


# cube for manipulation
target_cube = DynamicCuboid(
    prim_path="/World/target_cube",
    name="target_cube",
    position=cube_initial_position,
    size=0.04,
    color=np.array([1.0, 0.0, 0.0])
)
my_world.scene.add(target_cube)

stage = omni.usd.get_context().get_stage()
box_prim = stage.GetPrimAtPath("/World/target_marker")

# marker at the target position
target_marker = DynamicCuboid(
    prim_path="/World/target_marker",
    name="target_marker",
    position=target_position,
    size=0.04,
    color=np.array([1.0, 0.0, 0.0])
)
my_world.scene.add(target_marker)

stage = omni.usd.get_context().get_stage()
box_prim = stage.GetPrimAtPath("/World/target_marker")

# disable gravity for the marker
physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(box_prim)
physx_api.CreateDisableGravityAttr(True)

# disable collision for the marker
collision_api = UsdPhysics.CollisionAPI.Apply(box_prim)
collision_api.CreateCollisionEnabledAttr(False)

my_controller = PickPlaceController(name="controller", robot_articulation=my_fetch, gripper=my_fetch.gripper)
task_params = my_world.get_task("fetch_pick_place").get_params()
articulation_controller = my_fetch.get_articulation_controller()
i = 0
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
        observations = my_world.get_observations()

        actions = my_controller.forward(
            picking_position=cube_initial_position,
            placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_orientation=cube_initial_orientation,
            end_effector_offset=np.array([0, 0, 0]),
        )

        if my_controller.is_done():
            print("done picking and placing")
        articulation_controller.apply_action(actions)
    if args.test is True:
        break
simulation_app.close()
