"""[L8 완성] 색 분류 Pick & Place (Isaac Sim PC 쪽)

흐름:
  1) 파랑/초록 큐브 중 하나가 pick 영역 랜덤 위치에 스폰
  2) 로봇이 자동으로 큐브를 집어서 들어 올리고 멈춘다 (손목 카메라는 계속 /rgb 발행)
  3) 일반 노트북이 색을 감지해서 /color_id (std_msgs/Int32) 발행: 파랑=1, 초록=2
  4) /color_id 를 받으면 같은 색 마커 위에 놓고 관찰 자세로 복귀
  5) 초기화: 새 랜덤 큐브 스폰 -> 다시 2)

실행 (ros_set 안 한 새 터미널):
  isaac_ros
  isaac_python L8_color_sort.py     # 창이 뜨면 Play
테스트 (일반 노트북):
  ros2 topic pub --once /color_id std_msgs/msg/Int32 "{data: 1}"
"""

from pathlib import Path

# ============================================================
# 1. 앱 시작 + 확장 켜기 (씬에 ROS2 OmniGraph 가 있으므로 씬 열기 전에)
# ============================================================
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.robot_motion.motion_generation")
simulation_app.update()

# ============================================================
# 2. 모듈 import (rclpy 는 ROS2 Bridge 를 켠 뒤에)
# ============================================================
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.stage import get_current_stage, is_stage_loading, open_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

ROOT = Path(__file__).resolve().parents[1]
SCENE_USD = ROOT / "scenes" / "pick_place_cube_ros.usd"
URDF_PATH = ROOT / "robot" / "m0609_rg2.urdf"
DESCRIPTOR_PATH = ROOT / "descriptor" / "m0609_robot_description.yaml"
ROBOT_PRIM = "/World/m0609_rg2_d455"
EE_FRAME = "gripper_center"          # URDF 의 그리퍼 중심 프레임 (link_6 에서 z 0.185 m)

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
OBSERVE_DEG = [0, 0, 90, 0, 90, 0]   # 관찰(홈) 자세: 여기서 카메라로 큐브를 본다

# L7 에서 검증된 "큐브를 집는 자세". 이 자세의 그리퍼 방향(아래를 봄)과
# 큐브 대비 그리퍼 위치를 기준으로 삼아, 랜덤 위치에서도 같은 방식으로 집는다
L7_PICK_DEG = [0, 18, 101, 0, 61, 0]
L7_CUBE_POSITION = np.array([0.45, 0.0, 0.02575])

CUBE_SIZE = 0.0515
COLORS = ["blue", "green"]           # 인덱스+1 = color_id (파랑=1, 초록=2)
RGB = {"blue": np.array([0.0, 0.0, 1.0]), "green": np.array([0.0, 1.0, 0.0])}

PICK_X = (0.40, 0.50)                # pick 영역 (로봇 정면)
PICK_Y = (-0.08, 0.08)

MARKERS = {                          # 놓을 곳 (바닥 위 얇은 판)
    "blue": np.array([0.32, -0.32, 0.001]),
    "green": np.array([0.32, 0.32, 0.001]),
}
MARKER_SIZE = 0.08

PARK = {                             # 이번 판에 안 쓰는 큐브를 두는 곳 (로봇 뒤)
    "blue": np.array([-0.45, -0.25, CUBE_SIZE / 2]),
    "green": np.array([-0.45, 0.25, CUBE_SIZE / 2]),
}

APPROACH_HEIGHT = 0.12    # 집기/놓기 전에 이만큼 위에서 내려간다 (m)
PLACE_CLEARANCE = 0.005   # 놓을 때 바닥에서 살짝 띄운다 (m)
TOLERANCE_DEG = 1.0       # 팔 관절이 목표와 이만큼 안이면 도착
GRIPPER_FRAMES = 60       # 그리퍼 단계 대기 (60 프레임 = 1 초)
STEP_TIMEOUT_FRAMES = 600 # 팔 단계가 10 초 안에 도착 못하면 경고 후 다음 단계
SPAWN_SETTLE_FRAMES = 30 # 새 큐브 스폰 후 0.5 초 기다렸다가 집으러 간다 (큐브가 바닥에 안착)
HOLD_SETTLE_FRAMES = 60  # 들어 올린 뒤 1 초는 color_id 무시 (집는 도중에 온 값이 섞이지 않게)


# ------------------------------------------------------------
# 회전 도우미
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# (가) Task: 로봇 + 마커 2개 + 큐브 2개, 초기화마다 큐브 하나를 랜덤 스폰
# ------------------------------------------------------------
class ColorSortTask(BaseTask):
    def __init__(self, name):
        super().__init__(name)
        self._rng = np.random.default_rng()
        self.target_color = None   # 이번 판 큐브 색 (정답 비교용)

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
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
        for color, pos in MARKERS.items():
            scene.add(
                VisualCuboid(
                    prim_path=f"/World/Marker_{color}",
                    name=f"marker_{color}",
                    position=pos,
                    scale=np.array([MARKER_SIZE, MARKER_SIZE, 0.002]),
                    size=1.0,
                    color=RGB[color],
                )
            )
        self.cubes = {}
        for color in COLORS:
            self.cubes[color] = scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Cube_{color}",
                    name=f"cube_{color}",
                    position=PARK[color],
                    scale=np.array([CUBE_SIZE] * 3),
                    size=1.0,
                    color=RGB[color],
                )
            )
            self._task_objects[self.cubes[color].name] = self.cubes[color]
        self._task_objects[self._robot.name] = self._robot

    def post_reset(self):
        q = np.zeros(self._robot.num_dof)
        for name, angle in zip(ARM_JOINTS, np.radians(OBSERVE_DEG)):
            q[self._robot.get_dof_index(name)] = angle
        self._robot.set_joint_positions(q)
        self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
        self.spawn_random_cube()

    def spawn_random_cube(self):
        """큐브 하나를 골라 pick 영역 랜덤 위치로, 나머지는 대기 위치로"""
        self.target_color = str(self._rng.choice(COLORS))
        pick_pos = np.array([self._rng.uniform(*PICK_X), self._rng.uniform(*PICK_Y), CUBE_SIZE / 2])
        for color, cube in self.cubes.items():
            pos = pick_pos if color == self.target_color else PARK[color]
            cube.set_world_pose(position=pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        print(f"\n[spawn] {self.target_color} 큐브 -> x {pick_pos[0]:.3f} y {pick_pos[1]:.3f}"
              f" (정답 color_id = {COLORS.index(self.target_color) + 1}) | 자동으로 집으러 갑니다")

    def target_cube_position(self):
        return self.cubes[self.target_color].get_world_pose()[0]

    def get_observations(self):
        return {self._robot.name: {"joint_positions": self._robot.get_joint_positions()}}

    def get_params(self):
        return {"robot_name": {"value": self._robot.name}}


# ------------------------------------------------------------
# (나) ROS2 노드: /color_id 구독 (2단계에서 확인한 것 그대로)
# ------------------------------------------------------------
class ColorIdSubscriber(Node):
    def __init__(self):
        super().__init__("isaac_color_sort")
        self.latest_id = None
        self.create_subscription(Int32, "/color_id", self._callback, 10)
        self.get_logger().info("/color_id 구독 시작")

    def _callback(self, msg):
        if msg.data in (1, 2):
            self.latest_id = msg.data
        else:
            self.get_logger().warn(f"알 수 없는 color_id: {msg.data} (1=파랑, 2=초록만 사용)")

    def pop(self):
        value, self.latest_id = self.latest_id, None
        return value


# ------------------------------------------------------------
# (다) IK 도우미: 위치 -> 팔 관절 6개 (rad)
#      L7 집기 자세의 그리퍼 방향을 목표 방향(yaw)만큼 돌려서 항상 아래를 보며 집는다
# ------------------------------------------------------------
class PickPlacePlanner:
    def __init__(self):
        self.kin = LulaKinematicsSolver(
            robot_description_path=str(DESCRIPTOR_PATH),
            urdf_path=str(URDF_PATH),
        )
        ee_pos, self._ee_rot = self.kin.compute_forward_kinematics(EE_FRAME, np.radians(L7_PICK_DEG))
        # L7 에서 큐브를 잡았을 때 "큐브 중심 -> gripper_center" 차이 (yaw=0 기준)
        self._grasp_offset = np.asarray(ee_pos) - L7_CUBE_POSITION
        print(f"[IK] L7 집기 자세 gripper_center {np.round(ee_pos, 3)} | 큐브 대비 오프셋 {np.round(self._grasp_offset, 3)}")

    def set_base_pose(self, position, orientation):
        self.kin.set_robot_base_pose(position, orientation)

    def solve(self, object_center, lift, warm_start):
        """물체 중심이 object_center 일 때 잡는 그리퍼 자세 (+ lift 만큼 위) 의 관절각"""
        yaw = np.arctan2(object_center[1], object_center[0])
        target = object_center + rot_z(yaw) @ self._grasp_offset + np.array([0.0, 0.0, lift])
        quat = rot_to_quat(rot_z(yaw) @ self._ee_rot)
        q, ok = self.kin.compute_inverse_kinematics(EE_FRAME, target, quat, warm_start=warm_start)
        return np.asarray(q), ok

    def _solve_all(self, targets, warm):
        """[(이름, 물체 중심, 높이)] 를 차례로 IK. 실패하면 None"""
        joints = {}
        for name, center, lift in targets:
            q, ok = self.solve(center, lift, warm)
            if not ok:
                print(f"[IK] 실패: {name} (목표 {np.round(center, 3)}, +{lift} m)")
                return None
            joints[name] = q
            warm = q   # 다음 IK 는 이전 해에서 시작 -> 관절이 크게 뒤집히지 않는다
        return joints

    def plan_pick(self, cube_pos):
        """집기 단계: 큐브 위 -> 내려가기 -> 닫기 -> 들어 올리기"""
        j = self._solve_all([
            ("큐브 위로", cube_pos, APPROACH_HEIGHT),
            ("내려가기", cube_pos, 0.0),
            ("들어 올리기", cube_pos, APPROACH_HEIGHT),
        ], np.radians(L7_PICK_DEG))
        if j is None:
            return None
        return [
            ("큐브 위로", "arm", j["큐브 위로"]),
            ("내려가기", "arm", j["내려가기"]),
            ("그리퍼 닫기", "gripper", "close"),
            ("들어 올리기", "arm", j["들어 올리기"]),
        ]

    def plan_place(self, marker_pos, warm_start):
        """놓기 단계: 마커 위 -> 내려놓기 -> 열기 -> 팔 들기 -> 관찰 자세"""
        place_center = np.array([marker_pos[0], marker_pos[1], CUBE_SIZE / 2 + PLACE_CLEARANCE])
        j = self._solve_all([
            ("마커 위로", place_center, APPROACH_HEIGHT),
            ("내려놓기", place_center, 0.0),
            ("팔 들기", place_center, APPROACH_HEIGHT),
        ], warm_start)
        if j is None:
            return None
        return [
            ("마커 위로", "arm", j["마커 위로"]),
            ("내려놓기", "arm", j["내려놓기"]),
            ("그리퍼 열기", "gripper", "open"),
            ("팔 들기", "arm", j["팔 들기"]),
            ("관찰 자세 복귀", "arm", np.radians(OBSERVE_DEG)),
        ]


# ------------------------------------------------------------
# (라) Controller: 단계 목록을 하나씩 진행 (L7 StepController 와 같은 방식)
# ------------------------------------------------------------
class StepRunner:
    def __init__(self, gripper, arm_indices):
        self._gripper = gripper
        self._arm_indices = arm_indices
        self.steps = []
        self.step = 0
        self._frames = 0

    def start(self, steps):
        self.steps, self.step, self._frames = steps, 0, 0

    def is_done(self):
        return self.step >= len(self.steps)

    def forward(self, current_arm):
        name, kind, value = self.steps[self.step]
        self._frames += 1
        if kind == "arm":
            action = ArticulationAction(joint_positions=value, joint_indices=self._arm_indices)
            finished = np.max(np.abs(np.degrees(current_arm - value))) < TOLERANCE_DEG
            if not finished and self._frames >= STEP_TIMEOUT_FRAMES:
                print(f"  [경고] {name}: 시간 초과, 다음 단계로")
                finished = True
        else:
            action = self._gripper.forward(action=value)
            finished = self._frames >= GRIPPER_FRAMES
        if finished:
            print(f"  - {name} 완료")
            self.step += 1
            self._frames = 0
        return action


# ============================================================
# 3. 씬 열기 -> World -> Task / IK / ROS2 준비
# ============================================================
open_stage(str(SCENE_USD))
while is_stage_loading():
    simulation_app.update()

red_block = get_current_stage().GetPrimAtPath("/World/red_block")
if red_block.IsValid():
    red_block.SetActive(False)   # 이번 과제에 안 쓰는 빨간 블록

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
task = ColorSortTask(name="color_sort")
world.add_task(task)
world.reset()

robot = world.scene.get_object(task.get_params()["robot_name"]["value"])
arm = [robot.get_dof_index(name) for name in ARM_JOINTS]
hold_observe = ArticulationAction(joint_positions=np.radians(OBSERVE_DEG), joint_indices=arm)

planner = PickPlacePlanner()
planner.set_base_pose(*robot.get_world_pose())
runner = StepRunner(robot.gripper, arm)

rclpy.init()
color_node = ColorIdSubscriber()

# ============================================================
# 4. 메인 루프
#    SPAWN(안착 대기) -> PICK(자동 집기) -> HOLD(들고 /color_id 대기)
#    -> PLACE(마커에 놓기) -> 초기화 -> SPAWN ...
# ============================================================
def new_round():
    """새 랜덤 큐브를 스폰하고 SPAWN 상태로"""
    global state, settle
    task.spawn_random_cube()
    state, settle = "SPAWN", SPAWN_SETTLE_FRAMES


state = "SPAWN"
settle = SPAWN_SETTLE_FRAMES
reset_needed = False
place_color = None
hold_action = hold_observe   # HOLD 동안 유지할 팔 자세
round_no = 0

while simulation_app.is_running():
    world.step(render=True)
    rclpy.spin_once(color_node, timeout_sec=0.0)
    color_id = color_node.pop()   # 매 프레임 꺼낸다 (HOLD 가 아닐 때 온 값은 버려진다)

    if world.is_stopped() and not reset_needed:
        reset_needed = True
    if not world.is_playing():
        continue

    # Stop -> Play: 처음부터 (post_reset 에서 새 큐브 스폰)
    if reset_needed:
        world.reset()
        state, settle, reset_needed = "SPAWN", SPAWN_SETTLE_FRAMES, False
        continue

    current_arm = world.get_observations()[robot.name]["joint_positions"][arm]

    if state == "SPAWN":
        robot.apply_action(hold_observe)
        if settle > 0:
            settle -= 1
            continue
        steps = planner.plan_pick(task.target_cube_position())
        if steps is None:
            print("[IK] 집기 경로 계산 실패 -> 새 큐브로 초기화")
            new_round()
            continue
        runner.start(steps)
        state = "PICK"

    elif state == "PICK":
        robot.apply_action(runner.forward(current_arm))
        if runner.is_done():
            lifted = runner.steps[-1][2]   # 들어 올린 자세 (관절 rad)
            hold_action = ArticulationAction(joint_positions=lifted, joint_indices=arm)
            state, settle = "HOLD", HOLD_SETTLE_FRAMES
            print("[HOLD] 큐브를 들고 /color_id 대기 중 (1=파랑, 2=초록)")

    elif state == "HOLD":
        robot.apply_action(hold_action)   # 팔은 들어 올린 자세 유지 (그리퍼는 닫힌 명령 유지)
        if settle > 0:
            settle -= 1
            continue
        if color_id is None:
            continue

        place_color = COLORS[color_id - 1]
        answer = COLORS.index(task.target_color) + 1
        result = "정답" if color_id == answer else f"오답 (정답 {answer})"
        print(f"[color_id] {color_id} 수신 -> {place_color} 마커로 이동 | {result}")

        steps = planner.plan_place(MARKERS[place_color], warm_start=hold_action.joint_positions)
        if steps is None:
            print("[IK] 놓기 경로 계산 실패 -> 다른 값을 다시 보내 주세요")
            continue
        runner.start(steps)
        state = "PLACE"

    elif state == "PLACE":
        robot.apply_action(runner.forward(current_arm))
        if runner.is_done():
            round_no += 1
            placed = task.target_cube_position()
            dist = np.linalg.norm(placed[:2] - MARKERS[place_color][:2])
            print(f"[완료] {round_no}회차 | 큐브 -> {place_color} 마커 중심에서 {dist * 100:.1f} cm")
            new_round()   # 초기화: 관찰 자세 + 그리퍼 열린 상태이므로 큐브만 새로 스폰

# ============================================================
# 5. 종료
# ============================================================
color_node.destroy_node()
rclpy.shutdown()
simulation_app.close()
