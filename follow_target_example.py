from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse

import numpy as np
from controllers.rmpflow import RMPFlowController
from omni.isaac.core import World
# from ik_solver import KinematicsSolver
from tasks.follow_target import FollowTarget
# import carb

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

my_world = World(stage_units_in_meters=1.0)

my_task = FollowTarget(name="fetch_follow_target", target_position=np.array([1, 0, 0.5]))
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("fetch_follow_target").get_params()
target_name = task_params["target_name"]["value"]
fetch_name = task_params["robot_name"]["value"]
my_fetch = my_world.scene.get_object(fetch_name)

my_controller = RMPFlowController(name="target_follower_controller", robot_articulation=my_fetch)

ground_plane = my_world.scene.get_object(name="default_ground_plane")
my_controller.add_obstacle(ground_plane)

articulation_controller = my_fetch.get_articulation_controller()
reset_needed = False

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
        observations = my_world.get_observations()
        
        position = my_fetch.get_joint_positions()

        actions = my_controller.forward(
            target_end_effector_position=observations[target_name]["position"],
            target_end_effector_orientation=observations[target_name]["orientation"],
        )
        print("target end effector position:", observations[target_name]["orientation"])
        articulation_controller.apply_action(actions)
        
    if args.test is True:
        break
simulation_app.close()

# # ik solver
# my_controller = KinematicsSolver(my_fetch)
# articulation_controller = my_fetch.get_articulation_controller()

# while simulation_app.is_running():
#     my_world.step(render=True)
#     if my_world.is_playing():
#         if my_world.current_time_step_index == 0:
#             my_world.reset()
#         observations = my_world.get_observations()
#         actions, succ = my_controller.compute_inverse_kinematics(
#             target_position=observations[target_name]["position"],
#             target_orientation=observations[target_name]["orientation"],
#         )
#         if succ:
#             articulation_controller.apply_action(actions)
#         else:
#             carb.log_warn("IK did not converge to a solution.  No action is being taken.")
# simulation_app.close()