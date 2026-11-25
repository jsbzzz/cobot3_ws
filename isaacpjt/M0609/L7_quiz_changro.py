from pathlib import Path
  
    from isaacsim import SimulationApp
  
    simulation_app = SimulationApp({"headless": False})
  
    import numpy as np
    import omni.kit.commands
    from isaacsim.core.api import World
    from isaacsim.core.api.controllers import BaseController
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.api.tasks import BaseTask
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.robot.manipulators.grippers import ParallelGripper
    from isaacsim.robot.manipulators.manipulators import SingleManipulator
  
    ROBOT_USD = Path(__file__).resolve().parents[1] / "robot" / "m0609_rg2_d455.usd"
    ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    HOME_DEG = [0, 0, 90, 0, 90, 0]
    ABOVE_DEG = [0, 12, 83, 0, 85, 0]
    PICK_DEG = [0, 18, 101, 0, 61, 0]                                                                                          
    PLACE_ABOVE_DEG = [-45, 12, 83, 0, 85, 0]   # ★ -45 deg 방향 위 자세                                                       
    PLACE_DEG = [-45, 18, 101, 0, 61, 0]        # ★ -45 deg 방향 내려놓기 자세                                                 
    CUBE_POSITION = np.array([0.45, 0.0, 0.02575])
    CUBE_SIZE = 0.0515
    TOLERANCE_DEG = 1.0
    GRIPPER_FRAMES = 60
  
    # ★ pick(4단계) + place(4단계) = 총 8단계                                                                                  
    STEPS = [
        ("큐브 위로", "arm", ABOVE_DEG),                                                                                       
        ("내려가기", "arm", PICK_DEG),                                                                                         
        ("그리퍼 닫기", "gripper", "close"),                                                                                   
        ("들어 올리기", "arm", ABOVE_DEG),                                                                                     
        ("옆으로 돌기", "arm", PLACE_ABOVE_DEG),                                                                               
        ("내려놓기", "arm", PLACE_DEG),                                                                                        
        ("그리퍼 열기", "gripper", "open"),                                                                                    
        ("올라가기", "arm", PLACE_ABOVE_DEG),                                                                                  
    ]
  

    class PickEnvTask(BaseTask):
        def set_up_scene(self, scene):
            super().set_up_scene(scene)
            scene.add_default_ground_plane()
            add_reference_to_stage(
                usd_path=str(ROBOT_USD),
                prim_path="/World/m0609_rg2_d455",
            )
            gripper = ParallelGripper(
                end_effector_prim_path="/World/m0609_rg2_d455/gripper_body",
                joint_prim_names=["finger_joint"],
                joint_opened_positions=np.array([0.0]),
                joint_closed_positions=np.array([1.0]),
                use_mimic_joints=True,
            )
            self._robot = scene.add(
                SingleManipulator(
                    prim_path="/World/m0609_rg2_d455",
                    name="m0609",
                    end_effector_prim_path="/World/m0609_rg2_d455/gripper_body",
                    gripper=gripper,
                )
            )
            self._cube = scene.add(
                DynamicCuboid(
                    prim_path="/World/Cube",
                    name="cube",
                    position=CUBE_POSITION,
                    scale=np.array([CUBE_SIZE] * 3),
                    size=1.0,
                    color=np.array([0, 0, 1.0]),
                )
            )
            self._task_objects[self._robot.name] = self._robot
            self._task_objects[self._cube.name] = self._cube
  
        def post_reset(self): 
            home = np.zeros(self._robot.num_dof)
            for name, angle in zip(ARM_JOINTS, np.radians(HOME_DEG)):
                home[self._robot.get_dof_index(name)] = angle
            self._robot.set_joint_positions(home)
            self._robot.gripper.set_joint_positions(
                self._robot.gripper.joint_opened_positions,
            )
  
        def get_observations(self):
            return {
                self._robot.name: {"joint_positions": self._robot.get_joint_positions()},                                      
                self._cube.name: {"position": self._cube.get_world_pose()[0]},
            }
  
        def get_params(self):
            return {
                "robot_name": {"value": self._robot.name},
                "cube_name": {"value": self._cube.name},
            }
  

    class StepController(BaseController):
        def __init__(self, name, gripper, arm_indices):
            super().__init__(name)
            self._gripper = gripper
            self._arm_indices = arm_indices
            self.reset()
  
        def reset(self):
            self.step = 0
            self._step_frames = 0
  
        def is_done(self):
            return self.step >= len(STEPS)
  
        def forward(self, current_arm):
            if self.is_done():
                return None
            name, kind, value = STEPS[self.step]
            self._step_frames += 1 
  
            if kind == "arm": 
                action = ArticulationAction(
                    joint_positions=np.radians(value), 
                    joint_indices=self._arm_indices, 
                )
                finished = np.max(np.abs(np.degrees(current_arm) - value)) < TOLERANCE_DEG
            else:
                action = self._gripper.forward(action=value)
                finished = self._step_frames >= GRIPPER_FRAMES 
  
            if finished: 
                self.step += 1 
                self._step_frames = 0 
            return action
  
  
    world = World(
        stage_units_in_meters=1.0, 
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 60.0,
    )
    omni.kit.commands.execute("SetLightingMenuModeCommand", lighting_mode="Default") 
  
    task = PickEnvTask(name="pick_env") 
    world.add_task(task)
    world.reset()
  
    params = task.get_params()
    robot_name = params["robot_name"]["value"]
    cube_name = params["cube_name"]["value"]
    robot = world.scene.get_object(robot_name)
    arm = [robot.get_dof_index(name) for name in ARM_JOINTS] 
    controller = StepController("step_controller", gripper=robot.gripper, arm_indices=arm) 
  
    reset_needed = False
    last_step = 0
    while simulation_app.is_running():
        world.step(render=True)
  
        if world.is_stopped() and not reset_needed:
            reset_needed = True 
  
        if world.is_playing():
            if reset_needed:
                world.reset() 
                controller.reset()
                last_step = 0
                reset_needed = False
                print("reset 완료: 처음부터 다시 시작")                                                                        
  
            observations = world.get_observations()
            current_arm = observations[robot_name]["joint_positions"][arm]
            action = controller.forward(current_arm) 
            if action is not None:
                robot.apply_action(action)
  
            if controller.step != last_step: 
                finger_idx = robot.get_dof_index("finger_joint") 
                finger = observations[robot_name]["joint_positions"][finger_idx]
                x, y, z = observations[cube_name]["position"] 
                print(
                    f"시각 {world.current_time:5.2f} s | {STEPS[last_step][0]} 끝"                                             
                    f" | 그리퍼 {finger:.2f} rad | 큐브 x {x:5.2f} y {y:5.2f} z {z:.3f} m"                                     
                )
                last_step = controller.step
                if controller.is_done():
                    print("모든 단계 완료")                                                                                    
  
    simulation_app.close()
