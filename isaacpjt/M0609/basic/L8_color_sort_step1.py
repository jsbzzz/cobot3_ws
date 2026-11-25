"""[L8 1단계] 씬 불러오기 + 색 마커 + 랜덤 큐브 스폰 + 관찰 자세

실행: isaac_python L8_color_sort_step1.py   (Play 누르기)
확인: 일반 노트북에서  ros2 run rqt_image_view rqt_image_view  -> /rgb 선택
      pick 영역의 큐브(파랑 또는 초록)가 화면에 보이면 성공
      Stop -> Play 할 때마다 큐브 색/위치가 랜덤으로 바뀐다
"""

from pathlib import Path

# ============================================================
# 1. 앱 시작 + ROS2 Bridge 켜기 (씬에 ROS2 OmniGraph 가 있으므로 씬 열기 전에)
# ============================================================
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# ============================================================
# 2. Isaac 모듈 import
# ============================================================
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.stage import get_current_stage, is_stage_loading, open_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

ROOT = Path(__file__).resolve().parents[1]
SCENE_USD = ROOT / "scenes" / "pick_place_cube_ros.usd"   # 카메라 /rgb, /depth 그래프가 들어있는 씬
ROBOT_PRIM = "/World/m0609_rg2_d455"                       # 씬 안의 로봇 prim

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
OBSERVE_DEG = [0, 0, 90, 0, 90, 0]   # 관찰 자세: 손목 카메라가 pick 영역을 내려다보는 자세 (화면 보고 조정)

CUBE_SIZE = 0.0515
COLORS = ["blue", "green"]                                  # 인덱스+1 = color_id (파랑=1, 초록=2)
RGB = {"blue": np.array([0.0, 0.0, 1.0]), "green": np.array([0.0, 1.0, 0.0])}

# pick 영역 (로봇 정면). 이 범위 안에서 랜덤 위치
PICK_X = (0.40, 0.50)
PICK_Y = (-0.08, 0.08)

# 마커 위치: 큐브를 내려놓을 곳 (pick 영역 카메라 화면에 안 들어오게 옆으로 뺌)
MARKERS = {
    "blue": np.array([0.32, -0.32, 0.001]),   # 오른쪽 -45도
    "green": np.array([0.32, 0.32, 0.001]),   # 왼쪽 +45도
}
MARKER_SIZE = 0.08

# 대기 위치: pick 영역에 안 나간 큐브를 두는 곳 (로봇 뒤, 카메라에 안 보이게)
PARK = {
    "blue": np.array([-0.45, -0.25, CUBE_SIZE / 2]),
    "green": np.array([-0.45, 0.25, CUBE_SIZE / 2]),
}


# ------------------------------------------------------------
# (가) Task: 로봇 + 마커 2개 + 큐브 2개, 리셋마다 큐브 하나를 랜덤 스폰
# ------------------------------------------------------------
class ColorSortTask(BaseTask):
    def __init__(self, name):
        super().__init__(name)
        self._rng = np.random.default_rng()
        self.target_color = None   # 이번에 pick 영역에 나간 큐브 색 (정답, 디버깅용)

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        # 바닥/로봇은 씬 USD 에 이미 있으므로 추가하지 않는다

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

        # 색 마커: 충돌 없는 얇은 판 (눈으로 보이는 표시만)
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

        # 큐브 2개: 처음엔 둘 다 대기 위치
        self._cubes = {}
        for color in COLORS:
            self._cubes[color] = scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Cube_{color}",
                    name=f"cube_{color}",
                    position=PARK[color],
                    scale=np.array([CUBE_SIZE] * 3),
                    size=1.0,
                    color=RGB[color],
                )
            )
            self._task_objects[self._cubes[color].name] = self._cubes[color]
        self._task_objects[self._robot.name] = self._robot

    def post_reset(self):
        # 팔: 관찰 자세로 순간이동, 그리퍼: 열기
        q = np.zeros(self._robot.num_dof)
        for name, angle in zip(ARM_JOINTS, np.radians(OBSERVE_DEG)):
            q[self._robot.get_dof_index(name)] = angle
        self._robot.set_joint_positions(q)
        self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
        self.spawn_random_cube()

    def spawn_random_cube(self):
        """큐브 2개 중 하나를 골라 pick 영역 랜덤 위치로, 나머지는 대기 위치로"""
        self.target_color = str(self._rng.choice(COLORS))
        pick_pos = np.array([
            self._rng.uniform(*PICK_X),
            self._rng.uniform(*PICK_Y),
            CUBE_SIZE / 2,
        ])
        for color, cube in self._cubes.items():
            pos = pick_pos if color == self.target_color else PARK[color]
            cube.set_world_pose(position=pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        print(f"[spawn] {self.target_color} 큐브 -> x {pick_pos[0]:.3f} y {pick_pos[1]:.3f}"
              f" (정답 color_id = {COLORS.index(self.target_color) + 1})")

    def get_observations(self):
        obs = {self._robot.name: {"joint_positions": self._robot.get_joint_positions()}}
        for color, cube in self._cubes.items():
            obs[cube.name] = {"position": cube.get_world_pose()[0]}
        return obs

    def get_params(self):
        return {"robot_name": {"value": self._robot.name}}


# ------------------------------------------------------------
# (나) 씬 열기 -> World -> Task 등록
# ------------------------------------------------------------
open_stage(str(SCENE_USD))
while is_stage_loading():
    simulation_app.update()

# 씬에 있던 빨간 블록은 이번 과제에 안 쓰므로 비활성화
red_block = get_current_stage().GetPrimAtPath("/World/red_block")
if red_block.IsValid():
    red_block.SetActive(False)

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
task = ColorSortTask(name="color_sort")
world.add_task(task)
world.reset()

robot = world.scene.get_object(task.get_params()["robot_name"]["value"])
arm = [robot.get_dof_index(name) for name in ARM_JOINTS]
hold_observe = ArticulationAction(joint_positions=np.radians(OBSERVE_DEG), joint_indices=arm)

# ============================================================
# 3. 메인 루프: 관찰 자세 유지. Stop -> Play 하면 새 랜덤 큐브
# ============================================================
reset_needed = False
while simulation_app.is_running():
    world.step(render=True)

    if world.is_stopped() and not reset_needed:
        reset_needed = True
    if world.is_playing():
        if reset_needed:
            world.reset()          # post_reset -> spawn_random_cube
            reset_needed = False
        robot.apply_action(hold_observe)   # USD 기본 drive 목표 대신 관찰 자세 유지

simulation_app.close()
