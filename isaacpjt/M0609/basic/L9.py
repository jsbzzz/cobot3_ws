"""[L9 연습] 빨간 타겟의 위치와 자세를 매 프레임 IK로 추종한다.

실행: isaac_python L9.py
Stage에서 /World/Target을 선택하고 W(이동), E(회전) 도구로 조작한다.
"""
from pathlib import Path

# ============================================================
# 1. 앱 시작 (Isaac 모듈보다 먼저)
# ============================================================
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# ============================================================
# 2. Isaac 모듈 import (앱을 켠 뒤에)
# ============================================================
import numpy as np
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

M0609_DIR = Path(__file__).resolve().parents[1]
ROBOT_USD = M0609_DIR / "robot" / "m0609_rg2_d455.usd"
URDF_PATH = M0609_DIR / "robot" / "m0609_rg2.urdf"
DESCRIPTION_PATH = M0609_DIR / "descriptor" / "m0609_robot_description.yaml"

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
HOME_DEG = [0, 0, 90, 0, 90, 0]
HOME_RAD = np.radians(HOME_DEG)
TCP_FRAME = "gripper_center"
ROBOT_PRIM = "/World/m0609_rg2_d455"
TARGET_PRIM = "/World/Target"
TARGET_SIZE = 0.04  # 한 변 4 cm
MAX_JUMP_DEG = 60.0  # 현재 관절값과 너무 먼 IK 해는 사용하지 않는다

# ------------------------------------------------------------
# (가) Task: 환경 구성, 홈 자세, 타겟 위치/자세 관측
# ------------------------------------------------------------
class FollowTargetTask(BaseTask):
    def __init__(self, name, ik_solver):
        super().__init__(name=name)
        self._ik_solver = ik_solver

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        add_reference_to_stage(usd_path=str(ROBOT_USD), prim_path=ROBOT_PRIM)

        gripper = ParallelGripper(
            end_effector_prim_path=f"{ROBOT_PRIM}/gripper_body",
            joint_prim_names=["finger_joint"],
            joint_opened_positions=np.array([0.0]),
            joint_closed_positions=np.array([1.0]),
            use_mimic_joints=True,
        )
        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM,
                name="m0609",
                end_effector_prim_path=f"{ROBOT_PRIM}/gripper_body",
                gripper=gripper,
            )
        )

        # VisualCuboid: 중력/충돌 없이 GUI에서 이동과 회전 가능.
        # 정확한 시작 위치와 자세는 post_reset()에서 FK로 설정한다.
        self._target = scene.add(
            VisualCuboid(
                prim_path=TARGET_PRIM,
                name="target",
                position=np.array([0.0, 0.0, 0.5]),
                size=TARGET_SIZE,
                color=np.array([1.0, 0.0, 0.0]),
            )
        )
        self._task_objects[self._robot.name] = self._robot
        self._task_objects[self._target.name] = self._target

    def post_reset(self):
        # 1. 팔은 홈 자세, 그리퍼는 열린 상태로 초기화 (L7 방식)
        arm = [self._robot.get_dof_index(name) for name in ARM_JOINTS]
        home = np.zeros(self._robot.num_dof)
        home[arm] = HOME_RAD
        self._robot.set_joint_positions(home)
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )
        # 이전 실행에서 남은 팔 목표까지 홈으로 돌린다.
        self._robot.apply_action(
            ArticulationAction(joint_positions=HOME_RAD.copy(), joint_indices=arm)
        )
        self._robot.apply_action(self._robot.gripper.forward(action="open"))

        # 2. FK: 홈 관절 각도 -> TCP 위치와 자세 (월드 좌표계)
        # L8과 같은 고정형 로봇이며 USD 루트가 IK 베이스와 일치한다고 가정.
        base_pos, base_orient = self._robot.get_world_pose()
        self._ik_solver.set_robot_base_pose(base_pos, base_orient)
        tcp_pos, tcp_rot = self._ik_solver.compute_forward_kinematics(
            TCP_FRAME, HOME_RAD
        )
        tcp_orient = rot_matrices_to_quats(tcp_rot)
        # 홈 자세 자체가 아래를 향하는 L7/L8의 자세이므로 FK 자세를 그대로 사용.
        # TCP와 타겟의 위치뿐 아니라 회전도 처음부터 정확히 맞춘다.
        self._target.set_world_pose(position=tcp_pos, orientation=tcp_orient)
        print("홈 TCP:", np.round(tcp_pos, 4), "m")
        if tcp_rot[2, 2] > -0.99:
            print("[확인 필요] 홈 TCP의 +Z축이 바닥 방향이 아닙니다. URDF/홈 자세 확인")

    def get_observations(self):
        target_pos, target_orient = self._target.get_world_pose()
        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
            },
            self._target.name: {
                "position": target_pos,
                "orientation": target_orient,
            },
        }

    def get_params(self):
        return {
            "robot_name": {"value": self._robot.name},
            "target_name": {"value": self._target.name},
        }


# ------------------------------------------------------------
# (나) Controller: 매 프레임 IK, 팔 형태 유지, 실패하면 마지막 명령 유지
# ------------------------------------------------------------
class FollowTargetController(BaseController):
    def __init__(self, name, ik_solver, arm_indices):
        super().__init__(name=name)
        self._ik_solver = ik_solver
        self._arm_indices = arm_indices
        self.reset()

    def reset(self):
        self._last_joints = HOME_RAD.copy()
        self.is_following = False
        self.reason = "초기 상태"

    def _same_arm_shape(self, joints):
        # 홈은 joint_3=+90도, joint_5=+90도.
        # 3번(팔꿈치)과 5번(손목)의 양의 굽힘 방향을 유지한다.
        # 0/180도 특이 자세와 반대 굽힘 해는 사용하지 않는다.
        return bool(0.0 < joints[2] < np.pi and 0.0 < joints[4] < np.pi)

    def forward(self, target_pos, target_orient, current_arm):
        # L8과 달리 목표 위치/자세를 매 프레임 받고 IK를 매번 계산한다.
        # warm_start는 홈이 아니라 현재 팔의 6개 관절값(rad).
        joints, success = self._ik_solver.compute_inverse_kinematics(
            TCP_FRAME,
            target_pos,
            target_orient,
            warm_start=current_arm,
        )

        # 성공했더라도 다른 팔 형태로 뒤집히거나 갑자기 크게 움직이는 해는 거부.
        self.is_following = False
        if not success or joints is None:
            self.reason = "IK 해 없음"
        else:
            joints = np.asarray(joints, dtype=float)
            if joints.shape != (6,) or not np.all(np.isfinite(joints)):
                self.reason = "잘못된 IK 해"
            elif not self._same_arm_shape(joints):
                self.reason = "홈과 다른 팔꿈치/손목 형태"
            elif np.max(np.abs(np.degrees(joints - current_arm))) > MAX_JUMP_DEG:
                self.reason = "현재 자세에서 너무 먼 IK 해"
            else:
                self._last_joints = joints.copy()
                self.is_following = True
                self.reason = "IK 해 사용 가능"

        # 실패 시에도 None을 반환하지 않고 마지막 유효한 관절 목표를 계속 보낸다.
        # 팔 6개만 제어하므로 그리퍼는 열린 상태를 유지한다.
        return ArticulationAction(
            joint_positions=self._last_joints.copy(),
            joint_indices=self._arm_indices,
        )


# ------------------------------------------------------------
# (다) World, Task, Controller 등록 및 준비
# ------------------------------------------------------------
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 120.0,
    rendering_dt=1.0 / 60.0,
)
omni.kit.commands.execute("SetLightingMenuModeCommand", lighting_mode="Default")

ik_solver = LulaKinematicsSolver(
    robot_description_path=str(DESCRIPTION_PATH),
    urdf_path=str(URDF_PATH),
)
# L7/L8의 관절 배열과 IK 관절 순서가 같은지 확인한다.
if list(ik_solver.get_joint_names()) != ARM_JOINTS:
    raise RuntimeError(f"IK 관절 순서 확인 필요: {ik_solver.get_joint_names()}")

task = FollowTargetTask(name="follow_target", ik_solver=ik_solver)
world.add_task(task)
world.reset()

params = task.get_params()
robot_name = params["robot_name"]["value"]
target_name = params["target_name"]["value"]
robot = world.scene.get_object(robot_name)
arm = [robot.get_dof_index(name) for name in ARM_JOINTS]
controller = FollowTargetController("follow_controller", ik_solver, arm)

print("Stage에서 /World/Target을 선택한 뒤 W(이동), E(회전)로 조작하세요.")

# ============================================================
# 3. 메인 루프 (L7의 Stop -> Play 초기화 구조 유지)
# ============================================================
reset_needed = False
last_following = None

try:
    while simulation_app.is_running():
        world.step(render=True)

        if world.is_stopped() and not reset_needed:
            reset_needed = True

        if world.is_playing():
            if reset_needed:
                world.reset()       # 팔, 그리퍼, 빨간 타겟 초기화
                controller.reset()  # 마지막 관절 명령과 추종 상태 초기화
                last_following = None
                reset_needed = False
                print("reset 완료: 처음부터 다시 시작")

            # 1. 관측 (Task)
            observations = world.get_observations()
            current_arm = observations[robot_name]["joint_positions"][arm]
            target_pos = observations[target_name]["position"]
            target_orient = observations[target_name]["orientation"]

            # 2. 계산 (Controller)
            action = controller.forward(target_pos, target_orient, current_arm)

            # 3. 명령 (팔 6개)
            robot.apply_action(action)

            # 4. 따라감/기다림 상태가 바뀌는 순간에만 한 줄 출력
            if controller.is_following != last_following:
                state = "따라감" if controller.is_following else "기다림"
                print(f"시각 {world.current_time:5.2f} s | {state} | {controller.reason}")
                last_following = controller.is_following
finally:
    simulation_app.close()
