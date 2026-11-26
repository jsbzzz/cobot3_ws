"""[L8 연습] 큐브 위치 추적 + IK Pick & Place

초기 큐브: (0.45, 0.20), 내려놓기: (0.45, -0.25)
실행: isaac_python L8_ik_tracking.py
가상 큐브의 world pose를 직접 읽는다 (카메라/ROS2 추적 아님).
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
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

# ★ 추가: IK 계산기
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

M0609_DIR = Path(__file__).resolve().parents[1]
ROBOT_USD = M0609_DIR / "robot" / "m0609_rg2_d455.usd"
URDF_PATH = M0609_DIR / "robot" / "m0609_rg2.urdf"
DESCRIPTION_PATH = M0609_DIR / "descriptor" / "m0609_robot_description.yaml"
ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
HOME_DEG = [0, 0, 90, 0, 90, 0]
TCP_FRAME = "gripper_center"
CUBE_SIZE = 0.0515
CUBE_POSITION = np.array([0.45, 0.20, CUBE_SIZE / 2])
PLACE_POSITION = np.array([0.45, -0.25, CUBE_SIZE / 2 + 0.005])
APPROACH_HEIGHT = 0.12
TOLERANCE_DEG = 1.0
GRIPPER_FRAMES = 60

# ★ 추적 설정: 잡기 전까지 5 mm 이상 이동하면 IK를 갱신한다.
TRACK_DISTANCE = 0.005
TRACK_STABLE_FRAMES = 15  # 내려가는 단계에서 목표 도착 및 큐브 안정 확인
TRACK_STABLE_DISTANCE = 0.002  # 직전 프레임 대비 2 mm 이상 움직이면 대기

# ★ L7에서 사용한 집기 자세는 TCP 보정에만 사용한다.
# 4개의 작업 자세는 아래에서 모두 IK로 계산한다.
L7_PICK_DEG = [0, 18, 101, 0, 61, 0]
L7_CUBE_POSITION = np.array([0.45, 0.0, CUBE_SIZE / 2])

ik_solver = LulaKinematicsSolver(
    robot_description_path=str(DESCRIPTION_PATH),
    urdf_path=str(URDF_PATH),
)

# FK: L7 집기 자세의 TCP 위치와 회전행렬 계산
ref_pos, ref_rot = ik_solver.compute_forward_kinematics(
    TCP_FRAME, np.radians(L7_PICK_DEG)
)
grasp_offset = np.asarray(ref_pos) - L7_CUBE_POSITION
print("[FK] TCP:", np.round(ref_pos, 4), "offset:", np.round(grasp_offset, 4))

def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_to_quat(r):
    """3x3 회전행렬 -> 쿼터니언 [w, x, y, z]"""
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


# ★ 목표 큐브 위치와 들어 올릴 높이 -> 관절 각도(deg)
def calculate_ik(center, lift, start_deg):
    yaw = np.arctan2(center[1], center[0])
    rotation = rot_z(yaw)
    target = center + rotation @ grasp_offset + np.array([0.0, 0.0, lift])
    orientation = rot_to_quat(rotation @ ref_rot)
    joints, success = ik_solver.compute_inverse_kinematics(
        TCP_FRAME,
        target,
        orientation,
        warm_start=np.radians(start_deg),
    )
    if not success:
        raise RuntimeError(f"IK 계산 실패: {target}")
    print("[IK] TCP 목표:", np.round(target, 4), "관절:", np.round(np.degrees(joints), 2))
    return np.degrees(joints)


# ★ 단계 목록은 L7과 동일. IK 목표는 Controller에서 채운다.
# None은 아직 계산되지 않은 관절 목표 (직접 명령으로 보내지 않는다).
STEPS = [
    ("큐브 위로", "arm", None),
    ("내려가기", "arm", None),
    ("그리퍼 닫기", "gripper", "close"),
    ("들어 올리기", "arm", None),
    ("목표 위치 위로 이동", "arm", None),
    ("바닥으로 내려가기", "arm", None),
    ("그리퍼 열기", "gripper", "open"),
    ("팔 들어 올리기", "arm", None),
    ("홈 자세 복귀", "arm", HOME_DEG),
]


# ------------------------------------------------------------
# (가) Task: 로봇(팔 + 그리퍼)과 큐브가 놓인 환경
# ------------------------------------------------------------
class PickEnvTask(BaseTask):
    """장면(바닥 + M0609 + RG2 그리퍼 + 큐브)을 만들고, 시작 자세를 맞추고, 관측값을 돌려준다"""

    # 언제: 첫 world.reset() 때 한 번
    # 할 일: 장면에 물체 추가
    def set_up_scene(self, scene):
        super().set_up_scene(scene)                   # 부모 것을 먼저 (self.scene 을 저장한다)
        scene.add_default_ground_plane()              # 바닥

        # 로봇 USD 참조: 파일의 기본 prim(로봇)을 장면에 넣는다
        add_reference_to_stage(
            usd_path=str(ROBOT_USD),                  # 참조할 USD 파일 경로
            prim_path="/World/m0609_rg2_d455",        # 장면에 만들 prim 경로 (GUI 의 Add Reference 처럼 파일 이름)
        )

        # 그리퍼 객체: finger_joint 하나로 열고 닫는다
        gripper = ParallelGripper(
            end_effector_prim_path="/World/m0609_rg2_d455/gripper_body",  # 그리퍼 본체 prim (rigid body 여야 함)
            joint_prim_names=["finger_joint"],        # 구동 관절 이름 (RG2 는 하나)
            joint_opened_positions=np.array([0.0]),   # 열림 각도 (rad)
            joint_closed_positions=np.array([1.0]),   # 닫힘 각도 (rad). 큐브를 잡으면 그 전에 멈춘다
            use_mimic_joints=True,                    # 나머지 손가락은 USD 의 mimic 관절이 따라 움직임
        )

        # 로봇 객체: 팔과 그리퍼를 한 객체로 (SingleArticulation 에 gripper 가 붙은 것)
        self._robot = scene.add(
            SingleManipulator(
                prim_path="/World/m0609_rg2_d455",    # 로봇 prim 경로 (위에서 USD 를 참조해 넣은 곳)
                name="m0609",                         # 장면에서 찾을 때 쓰는 이름 (world.scene.get_object)
                end_effector_prim_path="/World/m0609_rg2_d455/gripper_body",  # 그리퍼 본체 prim (ParallelGripper 와 같게)
                gripper=gripper,                      # 위에서 만든 그리퍼 객체 -> robot.gripper 로 꺼냄
            )
        )

        # 큐브: 중력과 충돌이 있는 상자 (그리퍼로 집을 물체)
        self._cube = scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",              # Stage 에 만들 prim 경로
                name="cube",                          # 장면에서 찾을 때 쓰는 이름
                position=CUBE_POSITION,               # 큐브 중심 위치 [x, y, z] (m)
                scale=np.array([CUBE_SIZE] * 3),      # x, y, z 방향 크기 배율 [0.0515, 0.0515, 0.0515]
                size=1.0,                             # 기본 한 변 1 m. 실제 크기 = size x scale = 0.0515 m
                color=np.array([0, 0, 1.0]),          # 색 [R, G, B] (0 ~ 1). 파란색
            )
        )

        # 이 Task 에 속한 물체로 등록
        self._task_objects[self._robot.name] = self._robot
        self._task_objects[self._cube.name] = self._cube

    # 언제: 매 world.reset() 끝에 (Stop -> Play 때도)
    # 할 일: 시작 자세 맞추기 (팔은 홈 자세, 그리퍼는 열림으로 순간이동)
    def post_reset(self):
        home = np.zeros(self._robot.num_dof)          # 관절 12개 값을 담을 배열 (팔 6 + 그리퍼 6)
        for name, angle in zip(ARM_JOINTS, np.radians(HOME_DEG)):
            home[self._robot.get_dof_index(name)] = angle  # 관절 이름으로 몇 번째인지 찾아 홈 각도를 넣는다
        self._robot.set_joint_positions(home)         # 팔을 홈 자세로 순간이동

        # 그리퍼 열기: 그리퍼 객체가 finger_joint 를 열림 각도로 순간이동시킨다
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions,  # 그리퍼를 만들 때 넣은 열림 각도 [0.0]
        )

    # 언제: world.get_observations() 를 부를 때
    # 돌려줄 값: {물체 이름: {값 이름: 값}}
    def get_observations(self):
        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),  # 관절 12개의 지금 각도 (rad)
            },
            self._cube.name: {
                "position": self._cube.get_world_pose()[0],           # 큐브 중심 위치 [x, y, z] (m)
            },
        }

    # 언제: 메인 루프가 물체 이름을 알아낼 때 (task.get_params())
    # 돌려줄 값: {파라미터 이름: {"value": 값}}
    def get_params(self):
        return {
            "robot_name": {"value": self._robot.name},  # "m0609"
            "cube_name": {"value": self._cube.name},    # "cube"
        }


# ------------------------------------------------------------
# (나) Controller: 단계 목록(STEPS)을 하나씩 진행한다 (상태 머신)
#      지금 단계 번호를 기억하고, 단계가 끝나면 다음 번호로 넘어간다
# ------------------------------------------------------------
class StepController(BaseController):
    """STEPS 의 단계를 차례로 진행한다. 팔 단계는 관절 목표를, 그리퍼 단계는 그리퍼 명령을 돌려준다"""

    def __init__(self, name, gripper, arm_indices):
        super().__init__(name)                        # 컨트롤러 이름
        self._gripper = gripper          # 그리퍼 객체 (forward("open" / "close") 로 명령을 만들어 준다)
        self._arm_indices = arm_indices  # 명령을 보낼 관절이 관절 배열의 몇 번째인지 (팔 6개)
        self.reset()

    # 언제: 처음 만들 때, 그리고 메인 루프가 world.reset() 할 때
    # 할 일: 기억하는 상태를 처음 값으로
    def reset(self):
        self.step = 0
        self._step_frames = 0
        # ★ 추가: 각 작업의 IK 목표와 큐브 추적 상태 초기화
        self._steps = list(STEPS)
        self._target_cube = None
        self._last_cube = None
        self._stable_frames = 0
        self._pick_valid = False
        self._place_valid = False

    def is_done(self):
        return self.step >= len(self._steps)

    # ★ 추가: 목표 각도를 해당 단계에 등록
    def _set_goal(self, step, joints_deg):
        name, kind, _ = self._steps[step]
        self._steps[step] = (name, kind, joints_deg)

    # ★ 추가: 현재 큐브 위치로 접근/집기/들어 올리기 목표 재계산
    def _update_pick(self, cube_pos, current_arm):
        self._target_cube = cube_pos.copy()
        self._pick_valid = False
        self._place_valid = False
        self._stable_frames = 0
        self._step_frames = 0
        # 내려가는 도중 큐브가 옮겨지면 먼저 새 위치 위로 다시 이동
        self.step = 0
        try:
            above = calculate_ik(cube_pos, APPROACH_HEIGHT,
                                 np.degrees(current_arm))
            pick = calculate_ik(cube_pos, 0.0, above)
        except RuntimeError as error:
            print(f"[추적] {error} -> 큐브가 이동할 때까지 대기")
            return
        self._set_goal(0, above)
        self._set_goal(1, pick)
        self._set_goal(3, above)
        self._pick_valid = True
        print("[추적] 새 큐브 위치:", np.round(cube_pos, 4))

    # ★ 추가: 잡기 직전, 고정된 Place 위치로 이동할 자세 계산
    def _prepare_place(self):
        above_pick = self._steps[3][2]
        try:
            above = calculate_ik(PLACE_POSITION, APPROACH_HEIGHT, above_pick)
            place = calculate_ik(PLACE_POSITION, 0.0, above)
        except RuntimeError as error:
            print(f"[Place] {error} -> 큐브를 잡지 않고 대기")
            return False
        self._set_goal(4, above)
        self._set_goal(5, place)
        self._set_goal(7, above)
        return True

    def _hold_arm(self, current_arm):
        return ArticulationAction(joint_positions=current_arm.copy(),
                                  joint_indices=self._arm_indices)

    # ★ 변경: 현재 큐브 좌표도 받아서 잡기 전까지 추적한다.
    def forward(self, current_arm, cube_pos):
        if self.is_done():
            return None

        cube_pos = np.asarray(cube_pos, dtype=float)
        if self.step in (0, 1):
            if not np.all(np.isfinite(cube_pos)):
                self._stable_frames = 0
                return self._hold_arm(current_arm)

            moved = (self._target_cube is None or
                     np.linalg.norm(cube_pos - self._target_cube) >= TRACK_DISTANCE)
            if moved:
                self._update_pick(cube_pos, current_arm)

            stable = (self._last_cube is not None and
                      np.linalg.norm(cube_pos - self._last_cube) < TRACK_STABLE_DISTANCE)
            self._last_cube = cube_pos.copy()

            if not self._pick_valid:
                return self._hold_arm(current_arm)
        else:
            # 그리퍼 닫기부터는 목표 고정: 잡힌 큐브를 다시 추적하지 않음
            stable = True

        name, kind, value = self._steps[self.step]
        self._step_frames += 1

        # 기존 L7: 팔은 관절 목표에 도착하면 다음 단계
        if kind == "arm":
            action = ArticulationAction(
                joint_positions=np.radians(value),
                joint_indices=self._arm_indices,
            )
            finished = np.max(np.abs(np.degrees(current_arm) - value)) < TOLERANCE_DEG

            # ★ 큐브 위에서 내려간 뒤, 잠시 안정되었을 때만 잡기 시작
            if self.step == 1:
                if finished and stable:
                    self._stable_frames += 1
                else:
                    self._stable_frames = 0
                finished = self._stable_frames >= TRACK_STABLE_FRAMES
                if finished:
                    # 실패한 Place IK를 매 프레임 반복하지 않는다.
                    if not self._place_valid:
                        self._place_valid = self._prepare_place()
                        if not self._place_valid:
                            self._pick_valid = False
                    finished = self._place_valid
                    if finished:
                        print("[추적 종료] 그리퍼 닫기 -> 들어 올리고 운반")
        else:
            # 기존 L7: 그리퍼는 정해진 프레임 동안 열거나 닫기
            action = self._gripper.forward(action=value)
            finished = self._step_frames >= GRIPPER_FRAMES

        if finished:
            self.step += 1
            self._step_frames = 0
        return action


# ------------------------------------------------------------
# (다) 등록과 준비
# ------------------------------------------------------------
world = World(
    stage_units_in_meters=1.0,                        # 길이 단위: 1 = 1 m
    physics_dt=1.0 / 120.0,                           # 물리 계산 주기 (s). 한 프레임에 물리 계산 2 번 (L2 와 같음)
    rendering_dt=1.0 / 60.0,                          # 한 프레임 시간 (s)
)
# 조명: 화면 위 Lighting 메뉴에서 "Default" 를 고른 것과 같다
omni.kit.commands.execute(
    "SetLightingMenuModeCommand",                     # 실행할 명령 이름
    lighting_mode="Default",                          # 조명 모드
)
task = PickEnvTask(name="pick_env")                   # Task 이름
world.add_task(task)                                  # World 에 Task 등록
world.reset()  # 여기서 task.set_up_scene() -> task.post_reset() 이 불린다

params = task.get_params()
robot_name = params["robot_name"]["value"]            # "m0609"
cube_name = params["cube_name"]["value"]              # "cube"
robot = world.scene.get_object(robot_name)            # 이름으로 장면에서 로봇 객체를 찾는다
arm = [robot.get_dof_index(name) for name in ARM_JOINTS]  # 팔 관절 6개가 관절 배열(12개)의 몇 번째인지
controller = StepController(
    "step_controller",                                # 컨트롤러 이름
    gripper=robot.gripper,                            # 로봇에 붙은 그리퍼 객체
    arm_indices=arm,                                  # 팔 관절 6개가 몇 번째인지
)

# ============================================================
# 3. 메인 루프 (창이 열려 있는 동안 한 프레임씩 진행)
# ============================================================
reset_needed = False
last_step = 0  # 바로 전 프레임의 단계 번호 (단계가 바뀌면 출력)
while simulation_app.is_running():
    world.step(render=True)                           # 한 프레임 진행 (render=True: 화면도 그린다)

    # 1) Stop 을 감지해 둔다
    if world.is_stopped() and not reset_needed:
        reset_needed = True
    if world.is_playing():
        # 2) Stop 다음 첫 Play 프레임: 장면과 컨트롤러를 함께 처음으로
        if reset_needed:
            world.reset()
            controller.reset()
            last_step = 0
            reset_needed = False
            print("reset 완료: 처음부터 다시 시작")
        # 3) 관측 (Task)
        observations = world.get_observations()
        current_arm = observations[robot_name]["joint_positions"][arm]  # 12개 중 팔 6개만
        # 4) 계산 (Controller)
        # ★ 변경: Task에서 읽은 현재 큐브 위치를 Controller에 전달
        cube_pos = observations[cube_name]["position"]
        action = controller.forward(current_arm, cube_pos)
        # 5) 명령 (다 끝났으면 None: 마지막 목표 그대로 유지)
        if action is not None:
            robot.apply_action(action)

        if controller.step < last_step:  # ★ 추적 도중 접근 단계로 돌아간 경우
            last_step = controller.step
        elif controller.step != last_step:  # 단계가 끝났을 때 한 번 출력
            finger = observations[robot_name]["joint_positions"][robot.get_dof_index("finger_joint")]  # 그리퍼 각도 (rad)
            x, y, z = observations[cube_name]["position"]                                          # 큐브 위치 (m)
            print(f"시각 {world.current_time:5.2f} s | {STEPS[last_step][0]} 끝 | 그리퍼 {finger:.2f} rad | 큐브 x {x:5.2f} y {y:5.2f} z {z:.3f} m")
            last_step = controller.step
            if controller.is_done():
                print("모든 단계 완료")

# ============================================================
# 4. 종료
# ============================================================
simulation_app.close()
