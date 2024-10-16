from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import numpy as np
from controllers.pick_place import PickPlaceController
from controllers.nav_controller import NavigationController
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import Articulation, ArticulationSubset
from tasks.pick_place import PickPlace
from pxr import PhysxSchema, UsdPhysics
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.core.materials.physics_material import PhysicsMaterial
import redis
import threading
from collections import deque
import json
import time

TORSO_HEIGHT = 0.1
all_objs = []
physics_material = PhysicsMaterial(
    prim_path="/World/Physics_Materials/object_manipulate",
    dynamic_friction=1.0,
    static_friction=0.5,
    restitution=0.0,
)

import math

def compute_gripper_quaternion(yaw_angle):
    theta = math.radians(40)  
    half_theta = theta / 2.0

    half_yaw = yaw_angle / 2.0

    sin_half_theta = math.sin(half_theta)
    cos_half_theta = math.cos(half_theta)
    q_tilt = [
        cos_half_theta,     
        0.0,                
        sin_half_theta,     
        0.0                  
    ]

    sin_half_yaw = math.sin(half_yaw)
    cos_half_yaw = math.cos(half_yaw)
    q_yaw = [
        cos_half_yaw,   
        0.0,          
        0.0,           
        sin_half_yaw   
    ]

    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return [w, x, y, z]

    q_total = quaternion_multiply(q_yaw, q_tilt)

    return q_total


def world_to_robot_frame(point_world, robot_position, robot_yaw):
    point_rel = point_world - robot_position

    cos_theta = np.cos(-robot_yaw)
    sin_theta = np.sin(-robot_yaw)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])

    point_rel_xy = point_rel[:2]
    point_local_xy = rotation_matrix @ point_rel_xy

    point_local = np.array([point_local_xy[0], point_local_xy[1], point_rel[2]])

    return point_local

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
    return parser.parse_args()

def create_world():
    return World(stage_units_in_meters=1.0)

def create_task(world, cube_initial_position, target_position):
    task = PickPlace(
        name="fetch_pick_place",
        cube_initial_position=cube_initial_position,  # This is a dummy prop
        target_position=target_position  # This is a dummy prop
    )
    world.add_task(task)
    world.reset()
    return task

def create_objects_for_manipulate(world, cube_initial_position, name, color, size=0.06):
    if world.scene.get_object(name) is not None:
        print(f"Object with name '{name}' already exists. Skipping creation.")
        return
    # The cube needs to be manipulated

    prim_path = "/World/objects/" + name
    cube_init = DynamicCuboid(
        prim_path=prim_path,
        name=name,
        position=cube_initial_position,
        size=size,
        color=color,
        mass=0.01,
        density=64e-8,
        physics_material=physics_material
    )
    world.scene.add(cube_init)
    configure_object_physics(prim_path, False, True)

def create_objects_for_target_visualization(world, target_viz_position, name, color, size=0.06):
    if world.scene.get_object(name) is not None:
        print(f"Object with name '{name}' already exists. Skipping creation.")
        return
    # The cube's target pose (just for visualization purposes, does not have collision)
    prim_path = "/World/objects/" + name
    cube_target = DynamicCuboid(
        prim_path=prim_path,
        name=name,
        position=target_viz_position,
        size=size,
        scale=np.array([1.0, 1.0, 0.01]),
        color=color
    )
    world.scene.add(cube_target)
    configure_object_physics(prim_path, True, False)

def configure_object_physics(prim_path: str, disable_gravity: bool, enable_collision: bool):
    stage = omni.usd.get_context().get_stage()
    target_prim = stage.GetPrimAtPath(prim_path)

    physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(target_prim)
    physx_api.CreateDisableGravityAttr(disable_gravity)

    collision_api = UsdPhysics.CollisionAPI.Apply(target_prim)
    collision_api.CreateCollisionEnabledAttr(enable_collision)

def create_pick_place_controller(fetch_robot, operation_mode='pick_place'):
    return PickPlaceController(
        name="controller",
        robot_articulation=fetch_robot,
        gripper=fetch_robot.gripper,
        operation_mode=operation_mode,
        robot_name='fetch_robot'
    )

def create_navagation_controller():
    return NavigationController()

def create_scene(world):
    usd_path = "/home/joel/.local/share/ov/pkg/isaac-sim-4.1.0/projects/tabletop_rearrangement/assets/table_original.usdz"
    add_reference_to_stage(usd_path=usd_path, prim_path="/World/table")
    scale_factor = 1.2
    world.scene.add(XFormPrim(
        prim_path="/World/table",
        name="table",
        position=np.array([0.83 * scale_factor, 0, 0]),
        orientation=np.array([0.5, 0.5, 0.5, 0.5]),
        scale=np.array([0.0055 * scale_factor, 0.008, 0.0105 * scale_factor])
    ))
    configure_object_physics("/World/table", True, True)
    
    scene_path = "/home/joel/.local/share/ov/pkg/isaac-sim-4.1.0/projects/tabletop_rearrangement/assets/scene.usd"
    add_reference_to_stage(usd_path=scene_path, prim_path="/World/scene")
    world.scene.add(XFormPrim(
        prim_path="/World/scene",
        name="scene",
    ))

def redis_listener(message_queue):
    redis_host = 'prerender.redis.cache.windows.net' 
    redis_port = 6379  
    redis_password = 'll8JdqSG3mFmZ9ZZnqGOMjDuVxHGEcJiRAzCaNFhjkM=' 

    r = redis.Redis(host=redis_host, port=redis_port, password=redis_password)
    p = r.pubsub()
    p.subscribe('task_channel')
    for message in p.listen():
        if message['type'] == 'message':
            data = message['data']
            message_queue.put(data)

def process_message(message_data, world, nav_controller, message_queue):
    global task_in_progress
    global current_pick_place_controller
    message = json.loads(message_data)
    instruction = message.get('instruction')
    fetch_robot = world.scene.get_object("fetch_robot")
    current_position, current_orientation = fetch_robot.get_world_pose()
    if instruction == 'create_objects':
        objects = message.get('objects', [])
        for obj in objects:
            obj_type = obj.get('type')
            name = obj.get('name')
            all_objs.append(name)
            position = np.array(obj.get('position'))
            color = np.array(obj.get('color'))
            size = obj.get('size', 0.06)
            if obj_type == 'manipulate':
                create_objects_for_manipulate(world, position, name, color, size)
            elif obj_type == 'target_visualization':
                create_objects_for_target_visualization(world, position, name, color, size)
    elif instruction in ['pick', 'place', 'pick_place']:
        task_params = message.get('task_params', {})
        end_effector_orientation = task_params.get('end_effector_orientation')
        print(end_effector_orientation)
        if end_effector_orientation is None:
            current_yaw = quat_to_euler_angles(current_orientation)[2]
            end_effector_orientation = np.array(compute_gripper_quaternion(current_yaw))
            print("Using default end effector orientation: ", end_effector_orientation)
        else:
            end_effector_orientation = np.array(task_params.get('end_effector_orientation'))

        operation_mode = instruction

        if operation_mode == 'pick':
            picking_position_world = np.array(task_params.get('picking_position'))
            # picking_position_local = world_to_robot_frame(picking_position_world, current_position, current_yaw)
            picking_position_local = picking_position_world
            picking_position_local[2] -= TORSO_HEIGHT
            placing_position_local = None 
        elif operation_mode == 'place':
            placing_position_world = np.array(task_params.get('placing_position'))
            # placing_position_local = world_to_robot_frame(placing_position_world, current_position, current_yaw)   
            placing_position_local = placing_position_world
            placing_position_local[2] -= TORSO_HEIGHT
            picking_position_local = None
            initial_end_effector_position_world = task_params.get('initial_end_effector_position')
            if initial_end_effector_position_world is not None:
                initial_end_effector_position_world = np.array(initial_end_effector_position_world)
            else:
                initial_end_effector_position_world = fetch_robot.end_effector.get_world_pose()[0]
            # initial_end_effector_position_local = world_to_robot_frame(initial_end_effector_position_world, current_position, current_yaw)
            initial_end_effector_position_local = initial_end_effector_position_world
            initial_end_effector_position_local[2] -= TORSO_HEIGHT
        else: 
            picking_position_world = np.array(task_params.get('picking_position'))
            placing_position_world = np.array(task_params.get('placing_position'))
            # picking_position_local = world_to_robot_frame(picking_position_world, current_position, current_yaw)
            # placing_position_local = world_to_robot_frame(placing_position_world, current_position, current_yaw)
            picking_position_local = picking_position_world
            picking_position_local = placing_position_world
            picking_position_local[2] -= TORSO_HEIGHT
            placing_position_local[2] -= TORSO_HEIGHT

        pick_place_controller = create_pick_place_controller(fetch_robot, operation_mode=operation_mode)

        if operation_mode == 'pick':
            pick_place_controller.set_pick_parameters(picking_position_local, end_effector_orientation)
        elif operation_mode == 'place':
            pick_place_controller.set_place_parameters(placing_position_local, end_effector_orientation, initial_end_effector_position=initial_end_effector_position_local)
        else:
            pick_place_controller.set_pick_place_parameters(picking_position_local, placing_position_local, end_effector_orientation)

        current_pick_place_controller = pick_place_controller

        task_in_progress = True
    elif instruction == 'reset_arm':
        subset_names = [
            "shoulder_lift_joint",
        ] 
        robot_subset = ArticulationSubset(fetch_robot, subset_names)
        robot_subset.apply_action([-25.0]) # Need to modify
    elif instruction == 'move':
        if task_in_progress:
            print("Task already in progress, will process the new task after completion")
        else:
            task_params = message.get('task_params', {})
            target_position = np.array(task_params.get('target_position'))
            target_orientation = task_params.get('target_orientation')
            if target_orientation is not None:
                target_orientation = np.array(target_orientation)
            else:
                target_orientation = None
            print("New navigation task received:", target_position, target_orientation)
            nav_controller.set_target(target_position, target_orientation)
            task_in_progress = True
    elif instruction == 'clear_objects':
        for obj in all_objs:
            world.scene.remove_object(obj)
        print("All objects cleared from the environment.")
    elif instruction == 'get_tasks':
        tasks = message_queue.show_tasks()
        print("Current tasks in queue:")
        for task_message_data in tasks:
            task_message = json.loads(task_message_data)
            print(task_message)
    else:
        print("Unknown instruction:", instruction)

def main_loop(simulation_app, world, nav_controller, articulation_controller, args):
    global task_in_progress
    global current_pick_place_controller
    reset_needed = False
    fetch_robot = world.scene.get_object("fetch_robot")
    while simulation_app.is_running():
        world.step(render=True)
        if world.is_stopped() and not reset_needed:
            reset_needed = True
        if world.is_playing():
            if reset_needed:
                world.reset()
                if current_pick_place_controller:
                    current_pick_place_controller.reset()
                reset_needed = False
            # Check for messages
            if not message_queue.empty():
                message_data = message_queue.peek()  # Peek without consuming
                message = json.loads(message_data)
                instruction = message.get('instruction')

                immediate_instructions = [
                    'create_objects', 'clear_objects', 'reset_arm', 'get_tasks'
                ]

                if instruction in immediate_instructions:
                    # Process immediately
                    message_data = message_queue.get()
                    process_message(message_data, world, nav_controller, message_queue)
                elif not task_in_progress:
                    # Task is not in progress, can start a new task
                    message_data = message_queue.get()
                    process_message(message_data, world, nav_controller, message_queue)
                else:
                    # Task in progress and instruction requires waiting
                    pass  # Do not consume the message; wait
            # Proceed with task execution
            if task_in_progress and current_pick_place_controller and current_pick_place_controller.task_parameters_set:
                observations = world.get_observations()
                actions = current_pick_place_controller.forward(
                    observations=observations
                )
                articulation_controller.apply_action(actions)
                if current_pick_place_controller.is_done():
                    print(f"Done {current_pick_place_controller.operation_mode}")
                    task_in_progress = False
                    current_pick_place_controller.reset()
                    current_pick_place_controller.task_parameters_set = False
                    current_pick_place_controller = None
            elif task_in_progress and nav_controller.target_set:
                current_position, current_orientation = fetch_robot.get_world_pose()
                target_position = nav_controller.target_position
                target_orientation = nav_controller.target_orientation
                initial_position_world = fetch_robot.get_joint_positions()
                actions = nav_controller.forward(current_position, current_orientation, initial_position_world, 0.005)
                articulation_controller.apply_action(actions)
                current_yaw = quat_to_euler_angles(current_orientation)[2]
                distance_to_target = np.linalg.norm(target_position - current_position[:2])

                if distance_to_target < nav_controller.position_threshold + 0.02:
                    print("Position Matched")
                    if target_orientation is None or abs(current_yaw - target_orientation) <= nav_controller.yaw_threshold:
                        nav_controller.target_set = False
                        task_in_progress = False
                        zero_velocities = [0.0, 0.0] + [None] * (len(initial_position_world) - 2)
                        fetch_robot.apply_action(ArticulationAction(joint_velocities=zero_velocities))
                        print("Done navigation")
                    else:
                        print(abs(current_yaw - target_orientation))
                else:
                    print(distance_to_target)
            else:
                pass
            if args.test:
                break

args = parse_arguments()
world = create_world()

cube_initial_position = np.array([0.7, 0, 0.58])
target_position = np.array([0.9, 0, 0.58])

task = create_task(world, cube_initial_position, target_position)
fetch_robot = world.scene.get_object("fetch_robot")
create_scene(world)

# Set robot joint init pose
subset_names = [
    "torso_lift_joint",
] 
robot_subset = ArticulationSubset(fetch_robot, subset_names)
robot_subset.apply_action([TORSO_HEIGHT])

nav_controller = create_navagation_controller()

articulation_controller = fetch_robot.get_articulation_controller()

task_in_progress = False

class TaskQueue:
    def __init__(self):
        self.queue = deque()
        self.lock = threading.Lock()
    def put(self, item):
        with self.lock:
            self.queue.append(item)
    def get(self):
        with self.lock:
            if self.queue:
                return self.queue.popleft()
            else:
                return None
    def peek(self):
        with self.lock:
            if self.queue:
                return self.queue[0]
            else:
                return None
    def empty(self):
        with self.lock:
            return len(self.queue) == 0
    def show_tasks(self):
        with self.lock:
            return list(self.queue)

message_queue = TaskQueue()

listener_thread = threading.Thread(target=redis_listener, args=(message_queue,))
listener_thread.daemon = True
listener_thread.start()

current_pick_place_controller = False
main_loop(simulation_app, world, nav_controller, articulation_controller, args)

simulation_app.close()
