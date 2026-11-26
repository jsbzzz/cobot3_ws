"""[L9 심화 미션] 타겟 따라가기 (Follow Target).

- 뷰포트에서 빨간 타겟(/World/Target)을 끌고 돌리면 로봇 그리퍼 TCP가 실시간으로 따라간다.
- Task(BaseTask) 와 Controller(BaseController) 역할을 명확히 분리한다.
- 매 프레임 현재 자세 근처(warm_start=current_arm)에서 IK를 풀되,
  팔 형태(joint_3 > 0, joint_5 > 0; 홈 90-90 형태)가 뒤집히지 않도록 유지한다.
- 도달 불가하거나 유효한 해가 없으면 마지막 유효 명령을 유지하며 기다린다.
- Stop -> Play 시 로봇·타겟·컨트롤러 상태를 처음(홈 자세)으로 초기화한다.

실행: isaac_python L9_changro.py
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
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.rotations import rot_matrix_to_quat
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

# 로봇 및 타겟 파라미터
ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
HOME_DEG = [0, 0, 90, 0, 90, 0]
TCP_FRAME = "gripper_center"
TARGET_SIZE = 0.04  # 한 변 4 cm
MIN_POSTURE_RAD = np.radians(1.0)  # 홈(90-90) 팔 형태 유지 기준: joint_3 > 1 deg, joint_5 > 1 deg


# ============================================================
# (가) Task: 장면(바닥 + 로봇 + 빨간 타겟) 구성, 리셋, 관측
# ============================================================
class FollowTargetTask(BaseTask):
    def __init__(self, name, init_target_pos, init_target_ori):
        super().__init__(name=name)
        self._init_target_pos = np.array(init_target_pos, dtype=float)
        self._init_target_ori = np.array(init_target_ori, dtype=float)

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
        self._target = scene.add(
            VisualCuboid(
                prim_path="/World/Target",
                name="target",
                position=self._init_target_pos,
                orientation=self._init_target_ori,
                scale=np.array([TARGET_SIZE] * 3),
                size=1.0,
                color=np.array([1.0, 0.0, 0.0]),
            )
        )
        self._task_objects[self._robot.name] = self._robot
        self._task_objects[self._target.name] = self._target

    def post_reset(self):
        home = np.zeros(self._robot.num_dof)
        for name, angle in zip(ARM_JOINTS, np.radians(HOME_DEG)):
            home[self._robot.get_dof_index(name)] = angle
        self._robot.set_joint_positions(home)
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )
        self._target.set_world_pose(
            position=self._init_target_pos,
            orientation=self._init_target_ori,
        )

    def get_observations(self):
        target_pos, target_ori = self._target.get_world_pose()
        return {
            self._robot.name: {"joint_positions": self._robot.get_joint_positions()},
            self._target.name: {"position": target_pos, "orientation": target_ori},
        }

    def get_params(self):
        return {
            "robot_name": {"value": self._robot.name},
            "target_name": {"value": self._target.name},
        }


# ============================================================
# (나) Controller: 매 프레임 IK 계산 + 팔 형태(90-90) 유지
# ============================================================
class FollowTargetController(BaseController):
    def __init__(self, name, ik_solver, arm_indices, home_rad):
        super().__init__(name)
        self._ik_solver = ik_solver
        self._arm_indices = arm_indices
        self._home_rad = np.array(home_rad, dtype=float)
        self.reset()

    def reset(self):
        self._last_joints = self._home_rad.copy()
        self.is_following = True

    @staticmethod
    def _keeps_home_posture(joints_rad):
        """홈 자세(joint_3=90, joint_5=90)와 같은 팔 형태(뒤집히지 않음)인지 확인한다."""
        return bool(joints_rad[2] > MIN_POSTURE_RAD and joints_rad[4] > MIN_POSTURE_RAD)

    def forward(self, current_arm, target_pos, target_ori):
        # 1) 매 프레임 현재 자세 근처(warm_start=current_arm)에서 먼저 IK 계산
        joints, success = self._ik_solver.compute_inverse_kinematics(
            TCP_FRAME,
            target_pos,
            target_ori,
            warm_start=current_arm,
        )

        # 2) 해가 없거나 팔 형태(joint_3, joint_5 > 0)가 뒤집히면 홈 자세 기준으로 다시 계산
        if not (success and self._keeps_home_posture(joints)):
            joints, success = self._ik_solver.compute_inverse_kinematics(
                TCP_FRAME,
                target_pos,
                target_ori,
                warm_start=self._home_rad,
            )

        # 3) 유효한 해가 있으면 갱신(따라감), 없으면 마지막 명령 유지(기다림)
        if success and self._keeps_home_posture(joints):
            self._last_joints = joints
            self.is_following = True
        else:
            self.is_following = False

        return ArticulationAction(
            joint_positions=self._last_joints,
            joint_indices=self._arm_indices,
        )


# ============================================================
# (다) IK 초기화 및 World · Task · Controller 연결
# ============================================================
ik_solver = LulaKinematicsSolver(
    robot_description_path=DESCRIPTION_PATH,
    urdf_path=URDF_PATH,
)
home_rad = np.radians(HOME_DEG)
home_tcp_pos, home_tcp_rot = ik_solver.compute_forward_kinematics(TCP_FRAME, home_rad)
home_tcp_ori = rot_matrix_to_quat(home_tcp_rot)

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 120.0,
    rendering_dt=1.0 / 60.0,
)
omni.kit.commands.execute("SetLightingMenuModeCommand", lighting_mode="Default")

task = FollowTargetTask(
    name="follow_target_env",
    init_target_pos=home_tcp_pos,
    init_target_ori=home_tcp_ori,
)
world.add_task(task)
world.reset()

params = task.get_params()
robot_name = params["robot_name"]["value"]
target_name = params["target_name"]["value"]
robot = world.scene.get_object(robot_name)
arm = [robot.get_dof_index(name) for name in ARM_JOINTS]

controller = FollowTargetController(
    name="follow_target_controller",
    ik_solver=ik_solver,
    arm_indices=arm,
    home_rad=home_rad,
)

# ============================================================
# 3. 메인 루프 (Stop -> Play 리셋 + 따라감/기다림 전환 시 1회 출력)
# ============================================================
reset_needed = False
last_following = None

while simulation_app.is_running():
    world.step(render=True)

    if world.is_stopped() and not reset_needed:
        reset_needed = True

    if world.is_playing():
        if reset_needed:
            world.reset()
            controller.reset()
            last_following = None
            reset_needed = False
            print("reset 완료: 처음부터 다시 시작")

        observations = world.get_observations()
        current_arm = observations[robot_name]["joint_positions"][arm]
        target_pos = observations[target_name]["position"]
        target_ori = observations[target_name]["orientation"]

        action = controller.forward(current_arm, target_pos, target_ori)
        robot.apply_action(action)

        if controller.is_following != last_following:
            status = "따라감" if controller.is_following else "기다림"
            print(
                f"시각 {world.current_time:5.2f} s | {status}"
                f" | 타겟 x {target_pos[0]:.2f} y {target_pos[1]:5.2f} z {target_pos[2]:.2f} m"
                f" | joint_1 ~ 6 {np.round(np.degrees(current_arm)).astype(int)} deg"
            )
            last_following = controller.is_following

# ============================================================
# 4. 종료
# ============================================================
simulation_app.close()
