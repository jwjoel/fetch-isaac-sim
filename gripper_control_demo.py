from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

my_world = World(stage_units_in_meters=1.0)
assets_root_path = '/home/joel/Development/Research/fetch-issac-sim/assets'
asset_path = assets_root_path + "/fetch/fetch_new.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World")

gripper = ParallelGripper(
    end_effector_prim_path="/World/fetch/gripper_link",
    joint_prim_names=["l_gripper_finger_joint", "r_gripper_finger_joint"],
    joint_opened_positions=np.array([0, 0]),
    joint_closed_positions=np.array([0.05, 0.05]),
    action_deltas=np.array([0.05, 0.05]),
)

my_denso = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/fetch",
        name="fetch_robot",
        end_effector_prim_path="/World/fetch",
        end_effector_prim_name="gripper_link",
        gripper=gripper,
    )
)

joints_default_positions = np.zeros(14)
joints_default_positions[12] = 0.5
joints_default_positions[13] = 0.5
my_denso.set_joints_default_state(positions=joints_default_positions)
my_world.scene.add_default_ground_plane()
my_world.reset()

i = 0
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
        i += 1
        gripper_positions = my_denso.gripper.get_joint_positions()
        if i < 500:
            # close the gripper slowly
            my_denso.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] + 0.01, gripper_positions[1] + 0.01])
            )
        if i > 500:
            # open the gripper slowly
            my_denso.gripper.apply_action(
                ArticulationAction(joint_positions=[gripper_positions[0] - 0.01, gripper_positions[1] - 0.01])
            )
        if i == 1000:
            i = 0
    if args.test is True:
        break

simulation_app.close()
