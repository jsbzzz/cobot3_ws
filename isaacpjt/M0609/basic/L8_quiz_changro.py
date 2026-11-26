"""[L8 연습 문제] IK 를 사용한 Pick and Place.

- 큐브 pick 위치: (0.45, 0.20)
- 큐브 place 위치: (0.45, -0.25)
- 자세 4개(큐브 위 / 집기 / 놓을 위치 위 / 내려놓기)를 LulaKinematicsSolver 로 계산
- Task(BaseTask) 와 Controller(BaseController) 역할을 분리해 구성

실행: isaac_python L8_quiz_changro.py
"""

from pathlib import Path

# ============================================================
# 1. 앱 시작 (항상 Isaac 모듈보다 먼저 실행)
# ============================================================
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# ============================================================
# 2. Isaac 모듈 import
# ============================================================
import numpy as np
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

# 파일 경로
M0609_DIR = Path(__file__).resolve().parents[1]
ROBOT_USD = f"{M0609_DIR}/robot/m0609_rg2_d455.usd"
URDF_PATH = f"{M0609_DIR}/robot/m0609_rg2.urdf"
DESCRIPTION_PATH = f"{M0609_DIR}/descriptor/m0609_robot_description.yaml"

# 로봇 및 작업 파라미터
ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
HOME_DEG = [0, 0, 90, 0, 90, 0]
TCP_FRAME = "gripper_center"

CUBE_SIZE = 0.0515
PICK_XY = (0.45, 0.20)
PLACE_XY = (0.45, -0.25)
CUBE_POSITION = np.array([PICK_XY[0], PICK_XY[1], CUBE_SIZE / 2.0])

ABOVE_Z = 0.200
GRASP_Z = 0.041
# 바닥 방향 (roll=180, pitch=0) + 홈 자세(joint_6=0) 기준 정렬을 위해 yaw=180 사용
TARGET_ORIENT = euler_angles_to_quat(np.radians([180, 0, 180]))

TOLERANCE_DEG = 1.0  # 팔 도착 허용 오차 (도)
GRIPPER_FRAMES = 60  # 그리퍼 동작 대기 프레임 (60프레임 = 1초)


def solve_ik_pose(ik_solver, label, target_pos, target_orient, warm_start_rad):
    """목표 TCP 위치와 자세로 팔 관절 6개 각도(deg, rad)를 IK 로 계산한다."""
    joints_rad, success = ik_solver.compute_inverse_kinematics(
        TCP_FRAME,
        target_pos,
        target_orient,
        warm_start=warm_start_rad,
    )
    if not success:
        raise RuntimeError(f"IK 해 없음 ({label}): 목표 {target_pos}")

    joints_deg = np.degrees(joints_rad)
    print(
        f"IK | {label} | 목표 x {target_pos[0]:.2f} y {target_pos[1]:5.2f} z {target_pos[2]:.3f} m"
        f" -> joint_1 ~ 6 {np.round(joints_deg).astype(int)} deg"
    )
    return joints_deg, joints_rad


# ============================================================
# (가) Task: 환경(로봇 + 그리퍼 + 큐브) 구성, 리셋, 관측
# ============================================================
class PickEnvTask(BaseTask):
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        add_reference_to_stage(
            usd_path=ROBOT_USD,
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
            self._robot.gripper.joint_opened_positions
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


# ============================================================
# (나) Controller: IK 로 계산된 단계 목록을 순서대로 실행하는 상태 머신
# ============================================================
class StepController(BaseController):
    def __init__(self, name, gripper, arm_indices, steps):
        super().__init__(name)
        self._gripper = gripper
        self._arm_indices = arm_indices
        self._steps = steps
        self.reset()

    def reset(self):
        self.step = 0
        self._step_frames = 0

    def is_done(self):
        return self.step >= len(self._steps)

    def forward(self, current_arm):
        if self.is_done():
            return None

        _, kind, value = self._steps[self.step]
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


# ============================================================
# (다) IK 계산 + World · Task · Controller 연결
# ============================================================
ik_solver = LulaKinematicsSolver(
    robot_description_path=DESCRIPTION_PATH,
    urdf_path=URDF_PATH,
)

home_rad = np.radians(HOME_DEG)
pick_above_deg, pick_above_rad = solve_ik_pose(
    ik_solver, "큐브 위", np.array([PICK_XY[0], PICK_XY[1], ABOVE_Z]), TARGET_ORIENT, home_rad
)
pick_deg, _ = solve_ik_pose(
    ik_solver, "집기", np.array([PICK_XY[0], PICK_XY[1], GRASP_Z]), TARGET_ORIENT, pick_above_rad
)
place_above_deg, place_above_rad = solve_ik_pose(
    ik_solver, "놓을 위치 위", np.array([PLACE_XY[0], PLACE_XY[1], ABOVE_Z]), TARGET_ORIENT, home_rad
)
place_deg, _ = solve_ik_pose(
    ik_solver, "내려놓기", np.array([PLACE_XY[0], PLACE_XY[1], GRASP_Z]), TARGET_ORIENT, place_above_rad
)

STEPS = [
    ("큐브 위로", "arm", pick_above_deg),
    ("내려가기", "arm", pick_deg),
    ("그리퍼 닫기", "gripper", "close"),
    ("들어 올리기", "arm", pick_above_deg),
    ("옆으로 옮기기", "arm", place_above_deg),
    ("내려놓기", "arm", place_deg),
    ("그리퍼 열기", "gripper", "open"),
    ("올라가기", "arm", place_above_deg),
]

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
finger_idx = robot.get_dof_index("finger_joint")

controller = StepController(
    "step_controller",
    gripper=robot.gripper,
    arm_indices=arm,
    steps=STEPS,
)

# ============================================================
# 3. 메인 루프 (Stop -> Play 리셋 + 단계 완료 시 TCP/관절 로그 출력)
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

        observations = world.get_observations()
        current_arm = observations[robot_name]["joint_positions"][arm]
        action = controller.forward(current_arm)
        if action is not None:
            robot.apply_action(action)

        if controller.step != last_step:
            tcp, _ = ik_solver.compute_forward_kinematics(TCP_FRAME, current_arm)
            finger = observations[robot_name]["joint_positions"][finger_idx]
            x, y, z = observations[cube_name]["position"]
            print(
                f"시각 {world.current_time:5.2f} s | {STEPS[last_step][0]} 끝"
                f" | TCP x {tcp[0]:.2f} y {tcp[1]:5.2f} z {tcp[2]:.3f} m"
                f" -> joint_1 ~ 6 {np.round(np.degrees(current_arm)).astype(int)} deg"
                f" | 그리퍼 {finger:.2f} rad | 큐브 x {x:5.2f} y {y:5.2f} z {z:.3f} m"
            )
            last_step = controller.step
            if controller.is_done():
                print("모든 단계 완료")

# ============================================================
# 4. 종료
# ============================================================
simulation_app.close()
