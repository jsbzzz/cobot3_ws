from pathlib import Path
                                                                                                                               
    # 1. 앱 시작 (항상 Isaac 모듈보다 먼저 실행)                                                                               
    from isaacsim import SimulationApp
  
    simulation_app = SimulationApp({"headless": False})
  
    # 2. Isaac 모듈 import                                                                                                     
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
    ABOVE_DEG = [0, 12, 83, 0, 85, 0]    # 큐브 위(약 0.2m 높이) 자세                                                          
    PICK_DEG = [0, 18, 101, 0, 61, 0]    # 큐브 집는 높이 자세                                                                 
    CUBE_POSITION = np.array([0.45, 0.0, 0.02575])
    CUBE_SIZE = 0.0515
    TOLERANCE_DEG = 1.0  # 팔 도착 허용 오차 (도)                                                                              
    GRIPPER_FRAMES = 60  # 그리퍼 동작 대기 프레임 (60프레임 = 1초)                                                            
  
    # ★ [핵심 1] 단계 목록(상태 머신): (단계이름, 종류, 목표값)                                                                
    #    동작을 늘릴 때 Controller/루프를 안 고치고 여기 목록만 추가하면 됨!                                                   
    STEPS = [                                                                                                                  
        ("큐브 위로", "arm", ABOVE_DEG),                                                                                       
        ("내려가기", "arm", PICK_DEG),                                                                                         
        ("그리퍼 닫기", "gripper", "close"),                                                                                   
        ("들어 올리기", "arm", ABOVE_DEG),                                                                                     
    ]
  

    # ============================================================
    # (가) Task: 환경(로봇 + 그리퍼 + 큐브) 구성 & 관측                                                                        
    # ============================================================
    class PickEnvTask(BaseTask):
        def set_up_scene(self, scene):
            super().set_up_scene(scene)
            scene.add_default_ground_plane()
            add_reference_to_stage(usd_path=str(ROBOT_USD), prim_path="/World/m0609_rg2_d455")
  
            # ★ [핵심 2] 그리퍼 객체화: finger_joint 하나로 열고(0.0) 닫음(1.0)                                                
            #    use_mimic_joints=True -> 나머지 손가락 관절은 USD mimic 설정으로 자동 연동                                    
            gripper = ParallelGripper(
                end_effector_prim_path="/World/m0609_rg2_d455/gripper_body",
                joint_prim_names=["finger_joint"],
                joint_opened_positions=np.array([0.0]),
                joint_closed_positions=np.array([1.0]),
                use_mimic_joints=True,
            )
  
            # ★ [핵심 3] SingleManipulator = 기존 SingleArticulation + gripper                                                 
            #    팔과 그리퍼를 한 로봇 객체로 묶어 robot.gripper 로 바로 꺼내 쓸 수 있음                                       
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
  
        # 매 world.reset() 끝마다 자동 호출 (Stop -> Play 포함)                                                                
        def post_reset(self):
            home = np.zeros(self._robot.num_dof)
            for name, angle in zip(ARM_JOINTS, np.radians(HOME_DEG)):
                home[self._robot.get_dof_index(name)] = angle                                                                  
            self._robot.set_joint_positions(home)  # 팔: 홈 자세 순간이동                                                      
            # 그리퍼: 열림 각도(0.0)로 순간이동                                                                                
            self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
  
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
  

    # ============================================================
    # (나) Controller: STEPS 를 순서대로 실행하는 상태 머신                                                                    
    # ============================================================
    class StepController(BaseController):
        def __init__(self, name, gripper, arm_indices):
            super().__init__(name)
            self._gripper = gripper
            self._arm_indices = arm_indices
            self.reset()
  
        def reset(self):
            self.step = 0          # 현재 진행 중인 단계 번호                                                                  
            self._step_frames = 0  # 현재 단계에서 지난 프레임 수                                                              
  
        def is_done(self):
            return self.step >= len(STEPS)
  
        def forward(self, current_arm):
            if self.is_done():
                return None  # 모든 단계 완료 -> 새 명령 없음 (마지막 자세 유지)                                               
  
            name, kind, value = STEPS[self.step]
            self._step_frames += 1
  
            # ★ [핵심 4] 단계 종류별 명령 & 완료 조건이 다름!                                                                  
            if kind == "arm":
                # 팔 단계: joint_indices 로 팔 6개만 명령 (그리퍼 상태는 유지)                                                 
                action = ArticulationAction(
                    joint_positions=np.radians(value),
                    joint_indices=self._arm_indices,
                )
                # 완료 조건: 팔 관절 6개 모두 목표와 1도 이내로 도착                                                           
                finished = np.max(np.abs(np.degrees(current_arm) - value)) < TOLERANCE_DEG
            else:
                # 그리퍼 단계: finger_joint 만 명령 (팔 자세는 안 건드림!)                                                     
                action = self._gripper.forward(action=value)
                # 완료 조건: 큐브를 잡으면 1.0 rad 까지 못 닫히므로 각도 대신 '프레임 수(60)'로 판정                           
                finished = self._step_frames >= GRIPPER_FRAMES
  
            if finished:
                self.step += 1
                self._step_frames = 0
            return action
  

    # ============================================================
    # (다) World · Task · Controller 연결                                                                                      
    # ============================================================                                                             
    # ★ [핵심 5] 물리 주기를 1/120초로 촘촘하게 설정 -> 그리퍼가 큐브를 안정적으로 잡음                                        
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
  
    # ============================================================
    # 3. 메인 루프 (Stop -> Play 리셋 + 단계 완료 로그)                                                                        
    # ============================================================
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
  
            # 관측(Task) -> 계산(Controller) -> 실행(Robot)                                                                    
            observations = world.get_observations()
            current_arm = observations[robot_name]["joint_positions"][arm]
            action = controller.forward(current_arm)
            if action is not None:
                robot.apply_action(action)
  
            # 단계 번호가 바뀐 순간(= 방금 단계 완료)에만 1회 출력                                                             
            if controller.step != last_step:
                finger = observations[robot_name]["joint_positions"][robot.get_dof_index("finger_joint")]
                x, y, z = observations[cube_name]["position"]
                print(f"시각 {world.current_time:5.2f} s | {STEPS[last_step][0]} 끝 | 그리퍼 {finger:.2f} rad | 큐브 x {x:5.2f}
                y {y:5.2f} z {z:.3f} m")
                last_step = controller.step
                if controller.is_done():
                    print("모든 단계 완료")                                                                                    
  
    # 4. 종료                                                                                                                  
    simulation_app.close()
