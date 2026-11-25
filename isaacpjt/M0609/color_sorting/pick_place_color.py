"""[STA 미션 - PC A (Isaac Sim)] Pick & Place FSM + ROS2 색상 감지 연동.

역할 (STA 사양서):
  1. 파란/초록 큐브 2개 중 하나를 랜덤 위치(pick 영역)에 스폰
  2. 손목 카메라(D455) 영상을 ROS2 토픽 /rgb (sensor_msgs/Image) 로 발행
  3. PC B(m0609_color_detector)가 보낸 /color_id (std_msgs/Int32) 구독
     - color_id == 1 -> 파란 마커 위치에 Place
     - color_id == 2 -> 초록 마커 위치에 Place
  4. Lula IK + Pick & Place FSM 으로 자동 집기 및 색상별 분류 반복

실행 (PC A):
  echo $ROS_DOMAIN_ID
  isaac_ros
  cd ~/cobot3_ws/isaacpjt/M0609/color_sorting
  isaac_python pick_place_color.py
"""

from pathlib import Path
import random

# ============================================================
# 1. 앱 시작 & ROS2 Bridge 확장 활성화 (Isaac 모듈보다 먼저)
# ============================================================
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# ============================================================
# 2. Isaac Sim / ROS2 모듈 import
# ============================================================
import numpy as np
import omni.graph.core as og
import omni.kit.commands
import omni.usd
import rclpy
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.rotations import rot_matrix_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)
from pxr import UsdGeom
from rclpy.node import Node
from std_msgs.msg import Int32

# ============================================================
# 3. 경로 및 파라미터 설정
# ============================================================
M0609_DIR = Path(__file__).resolve().parents[1]
ROBOT_USD = M0609_DIR / "robot" / "m0609_rg2_d455.usd"
ROBOT_URDF = M0609_DIR / "robot" / "m0609_rg2.urdf"
ROBOT_DESC_YAML = M0609_DIR / "descriptor" / "m0609_robot_description.yaml"

ROBOT_PRIM = "/World/m0609_rg2_d455"
EE_PRIM = f"{ROBOT_PRIM}/gripper_body"
EE_FRAME = "gripper_center"
DEFAULT_CAMERA_PRIM = (
    f"{ROBOT_PRIM}/angle_bracket/realsense_d455/RSD455/Camera_OmniVision_OV9782_Color"
)

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
HOME_DEG = [0, 0, 90, 0, 90, 0]

# 큐브 크기 및 랜덤 스폰(Pick) 영역 (로봇 도달 가능 범위)
CUBE_SIZE = 0.0515
CUBE_Z = CUBE_SIZE / 2.0 + 0.001
PICK_X_RANGE = (0.36, 0.48)
PICK_Y_RANGE = (-0.14, 0.14)

# 색상별 Place 마커 위치 (로봇에서 약 0.45m 거리)
COLOR_BLUE = 1
COLOR_GREEN = 2
BLUE_MARKER_XY = np.array([0.25, 0.36])
GREEN_MARKER_XY = np.array([0.25, -0.36])

# 대기 큐브 임시 보관 위치 (카메라 시야 밖)
PARK_BLUE_POS = np.array([-0.55, 0.35, CUBE_Z])
PARK_GREEN_POS = np.array([-0.55, -0.35, CUBE_Z])

# Pick & Place 높이 및 제어 허용 오차
OBSERVE_Z = 0.28       # 카메라로 큐브 색상을 관찰하는 높이
ABOVE_Z = 0.20         # 이동 / 들어 올리기 높이
PICK_Z = 0.028         # 파지 높이 (gripper_center 기준)
PLACE_Z = 0.038        # 내려놓기 높이
TOLERANCE_DEG = 1.5    # 팔 관절 도착 허용 오차 (deg)
GRIPPER_FRAMES = 60    # 그리퍼 열기/닫기 대기 프레임 (1초)
OBSERVE_SETTLE = 20    # 관찰 위치 도착 후 카메라 안정화 프레임


# ============================================================
# 4. ROS2 구독 노드: PC B 의 /color_id 수신
# ============================================================
class ColorIdSubscriber(Node):
    """PC B(m0609_color_detector)가 발행하는 /color_id (1=파랑, 2=초록)를 수신."""

    def __init__(self):
        super().__init__("m0609_pick_place_node")
        self.detected_color_id = None
        self.sub = self.create_subscription(Int32, "/color_id", self._callback, 10)

    def _callback(self, msg: Int32):
        if msg.data in (COLOR_BLUE, COLOR_GREEN):
            self.detected_color_id = int(msg.data)

    def clear(self):
        self.detected_color_id = None


# ============================================================
# 5. 손목 카메라 ROS2 Action Graph (/rgb, /clock 발행)
# ============================================================
def find_wrist_camera_prim() -> str:
    """Stage 에서 손목 컬러 카메라 prim 경로를 찾는다."""
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(DEFAULT_CAMERA_PRIM).IsValid():
        return DEFAULT_CAMERA_PRIM

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera) and "OV9782_Color" in prim.GetName():
            return str(prim.GetPath())
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Camera) and path.startswith(ROBOT_PRIM):
            return path
    raise RuntimeError(f"손목 카메라 prim 을 찾을 수 없습니다: {DEFAULT_CAMERA_PRIM}")


def setup_ros2_camera_and_clock_graph(camera_prim: str):
    """OmniGraph 로 /rgb (640x640) 와 /clock 토픽 발행 그래프를 생성한다."""
    try:
        import usdrt.Sdf
        cam_target = [usdrt.Sdf.Path(camera_prim)]
    except Exception:
        cam_target = [camera_prim]

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/World/ROS2_Graph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("CameraHelperRGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                ("ROS2Context.inputs:useDomainIDEnvVar", True),
                ("PublishClock.inputs:topicName", "/clock"),
                ("RenderProduct.inputs:cameraPrim", cam_target),
                ("RenderProduct.inputs:width", 640),
                ("RenderProduct.inputs:height", 640),
                ("CameraHelperRGB.inputs:topicName", "/rgb"),
                ("CameraHelperRGB.inputs:type", "rgb"),
                ("CameraHelperRGB.inputs:frameId", "sim_camera"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ROS2Context.outputs:context", "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("OnPlaybackTick.outputs:tick", "RenderProduct.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "CameraHelperRGB.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "CameraHelperRGB.inputs:renderProductPath"),
                ("ROS2Context.outputs:context", "CameraHelperRGB.inputs:context"),
            ],
        },
    )


# ============================================================
# 6. Task: 로봇 + 파란/초록 큐브 + 색상별 마커
# ============================================================
class ColorSortingTask(BaseTask):
    """M0609, 파란/초록 큐브 2개, 파란/초록 Place 마커 환경을 구성한다."""

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        add_reference_to_stage(usd_path=str(ROBOT_USD), prim_path=ROBOT_PRIM)

        gripper = ParallelGripper(
            end_effector_prim_path=EE_PRIM,
            joint_prim_names=["finger_joint"],
            joint_opened_positions=np.array([0.0]),
            joint_closed_positions=np.array([1.0]),
            use_mimic_joints=True,
        )
        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM,
                name="m0609",
                end_effector_prim_path=EE_PRIM,
                gripper=gripper,
            )
        )

        # 파란 큐브 & 초록 큐브
        self._blue_cube = scene.add(
            DynamicCuboid(
                prim_path="/World/BlueCube",
                name="blue_cube",
                position=PARK_BLUE_POS,
                scale=np.array([CUBE_SIZE] * 3),
                size=1.0,
                color=np.array([0.0, 0.2, 0.9]),
            )
        )
        self._green_cube = scene.add(
            DynamicCuboid(
                prim_path="/World/GreenCube",
                name="green_cube",
                position=PARK_GREEN_POS,
                scale=np.array([CUBE_SIZE] * 3),
                size=1.0,
                color=np.array([0.1, 0.75, 0.2]),
            )
        )

        # 색상별 Place 마커 (충돌 없는 VisualCuboid)
        scene.add(
            VisualCuboid(
                prim_path="/World/BlueMarker",
                name="blue_marker",
                position=np.array([BLUE_MARKER_XY[0], BLUE_MARKER_XY[1], 0.002]),
                scale=np.array([0.12, 0.12, 0.004]),
                size=1.0,
                color=np.array([0.0, 0.2, 0.9]),
            )
        )
        scene.add(
            VisualCuboid(
                prim_path="/World/GreenMarker",
                name="green_marker",
                position=np.array([GREEN_MARKER_XY[0], GREEN_MARKER_XY[1], 0.002]),
                scale=np.array([0.12, 0.12, 0.004]),
                size=1.0,
                color=np.array([0.1, 0.75, 0.2]),
            )
        )

        self._task_objects[self._robot.name] = self._robot
        self._task_objects[self._blue_cube.name] = self._blue_cube
        self._task_objects[self._green_cube.name] = self._green_cube

    def post_reset(self):
        home = np.zeros(self._robot.num_dof)
        for name, angle in zip(ARM_JOINTS, np.radians(HOME_DEG)):
            home[self._robot.get_dof_index(name)] = angle
        self._robot.set_joint_positions(home)
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )

    def spawn_random_cube(self):
        """파란/초록 큐브 중 하나를 랜덤 pick 위치로 옮기고 나머지는 대기 위치로 치운다."""
        pick_x = random.uniform(*PICK_X_RANGE)
        pick_y = random.uniform(*PICK_Y_RANGE)
        pick_pos = np.array([pick_x, pick_y, CUBE_Z])

        chosen_is_blue = random.choice([True, False])
        if chosen_is_blue:
            active, parked, park_pos = self._blue_cube, self._green_cube, PARK_GREEN_POS
            label = "파랑(예상=1)"
        else:
            active, parked, park_pos = self._green_cube, self._blue_cube, PARK_BLUE_POS
            label = "초록(예상=2)"

        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        active.set_world_pose(position=pick_pos, orientation=identity_quat)
        active.set_linear_velocity(np.zeros(3))
        active.set_angular_velocity(np.zeros(3))

        parked.set_world_pose(position=park_pos, orientation=identity_quat)
        parked.set_linear_velocity(np.zeros(3))
        parked.set_angular_velocity(np.zeros(3))

        print(f"[스폰] {label} 큐브 -> ({pick_x:.3f}, {pick_y:.3f}, {CUBE_Z:.3f})")
        return active, pick_pos

    def get_observations(self):
        return {
            self._robot.name: {"joint_positions": self._robot.get_joint_positions()},
            self._blue_cube.name: {"position": self._blue_cube.get_world_pose()[0]},
            self._green_cube.name: {"position": self._green_cube.get_world_pose()[0]},
        }

    def get_params(self):
        return {"robot_name": {"value": self._robot.name}}


# ============================================================
# 7. Pick & Place FSM 컨트롤러 (Lula IK + ROS2 /color_id 연동)
# ============================================================
class ColorPickPlaceFSM:
    """상태 머신으로 랜덤 큐브 관찰 -> 색상 수신 -> Pick -> 색상별 Place 를 반복."""

    def __init__(self, robot, task, arm_indices, ik_solver, down_quat, ros_node):
        self.robot = robot
        self.task = task
        self.arm_indices = arm_indices
        self.ik_solver = ik_solver
        self.down_quat = down_quat
        self.ros_node = ros_node
        self.reset()

    def reset(self):
        self.state = "SPAWN"
        self.state_frames = 0
        self.active_cube = None
        self.pick_xy = np.array([0.42, 0.0])
        self.place_xy = BLUE_MARKER_XY.copy()
        self.target_arm_rad = np.radians(HOME_DEG)
        self.ros_node.clear()

    def _set_ik_target(self, pos_xyz: np.ndarray) -> bool:
        action, success = self.ik_solver.compute_inverse_kinematics(
            target_position=pos_xyz,
            target_orientation=self.down_quat,
        )
        if not success or action.joint_positions is None:
            print(f"[경고] IK 실패: target={np.round(pos_xyz, 3)}")
            return False
        self.target_arm_rad = np.array(action.joint_positions, dtype=float)
        return True

    def _arm_arrived(self, current_arm_rad: np.ndarray) -> bool:
        err_deg = np.max(np.abs(np.degrees(current_arm_rad - self.target_arm_rad)))
        return bool(err_deg < TOLERANCE_DEG)

    def _arm_action(self) -> ArticulationAction:
        return ArticulationAction(
            joint_positions=self.target_arm_rad,
            joint_indices=self.arm_indices,
        )

    def _enter(self, next_state: str):
        self.state = next_state
        self.state_frames = 0

    def step(self, current_arm_rad: np.ndarray):
        self.state_frames += 1

        if self.state == "SPAWN":
            self.active_cube, pick_pos = self.task.spawn_random_cube()
            self.pick_xy = pick_pos[:2].copy()
            self.ros_node.clear()
            if self._set_ik_target(np.array([self.pick_xy[0], self.pick_xy[1], OBSERVE_Z])):
                self._enter("MOVE_OBSERVE")
            return self.robot.gripper.forward(action="open")

        if self.state == "MOVE_OBSERVE":
            if self._arm_arrived(current_arm_rad):
                self.ros_node.clear()
                print("[FSM] 관찰 위치 도착 -> PC B /color_id 대기 중...")
                self._enter("WAIT_COLOR")
            return self._arm_action()

        if self.state == "WAIT_COLOR":
            # 카메라 안정화 후 PC B 에서 받은 color_id 확인
            if self.state_frames >= OBSERVE_SETTLE and self.ros_node.detected_color_id is not None:
                cid = self.ros_node.detected_color_id
                if cid == COLOR_BLUE:
                    self.place_xy = BLUE_MARKER_XY.copy()
                    print("[FSM] /color_id=1 (파랑) 수신 -> 파란 마커로 이동 예정")
                else:
                    self.place_xy = GREEN_MARKER_XY.copy()
                    print("[FSM] /color_id=2 (초록) 수신 -> 초록 마커로 이동 예정")

                # 큐브 실제 현재 XY 로 정밀 보정
                cube_pos = self.active_cube.get_world_pose()[0]
                self.pick_xy = cube_pos[:2].copy()
                self._set_ik_target(np.array([self.pick_xy[0], self.pick_xy[1], PICK_Z]))
                self._enter("PICK_DOWN")
            return self._arm_action()

        if self.state == "PICK_DOWN":
            if self._arm_arrived(current_arm_rad):
                self._enter("GRIP_CLOSE")
            return self._arm_action()

        if self.state == "GRIP_CLOSE":
            if self.state_frames >= GRIPPER_FRAMES:
                self._set_ik_target(np.array([self.pick_xy[0], self.pick_xy[1], ABOVE_Z]))
                self._enter("LIFT")
            return self.robot.gripper.forward(action="close")

        if self.state == "LIFT":
            if self._arm_arrived(current_arm_rad):
                self._set_ik_target(np.array([self.place_xy[0], self.place_xy[1], ABOVE_Z]))
                self._enter("PLACE_ABOVE")
            return self._arm_action()

        if self.state == "PLACE_ABOVE":
            if self._arm_arrived(current_arm_rad):
                self._set_ik_target(np.array([self.place_xy[0], self.place_xy[1], PLACE_Z]))
                self._enter("PLACE_DOWN")
            return self._arm_action()

        if self.state == "PLACE_DOWN":
            if self._arm_arrived(current_arm_rad):
                self._enter("GRIP_OPEN")
            return self._arm_action()

        if self.state == "GRIP_OPEN":
            if self.state_frames >= GRIPPER_FRAMES:
                self._set_ik_target(np.array([self.place_xy[0], self.place_xy[1], ABOVE_Z]))
                self._enter("RETREAT")
            return self.robot.gripper.forward(action="open")

        if self.state == "RETREAT":
            if self._arm_arrived(current_arm_rad):
                self.target_arm_rad = np.radians(HOME_DEG)
                self._enter("RETURN_HOME")
            return self._arm_action()

        if self.state == "RETURN_HOME":
            if self._arm_arrived(current_arm_rad):
                print("[FSM] Pick & Place 1회 완료! 다음 큐브 스폰\n")
                self._enter("SPAWN")
            return self._arm_action()

        return None


# ============================================================
# 8. World · Task · ROS2 Graph · Lula IK 초기화
# ============================================================
for required_file in (ROBOT_USD, ROBOT_URDF, ROBOT_DESC_YAML):
    if not required_file.exists():
        raise FileNotFoundError(
            f"PC A 에 필수 파일이 없습니다: {required_file}\n"
            "M0609/robot/ 및 M0609/descriptor/ 파일 위치를 확인하세요."
        )

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 120.0,
    rendering_dt=1.0 / 60.0,
)
omni.kit.commands.execute("SetLightingMenuModeCommand", lighting_mode="Default")

task = ColorSortingTask(name="color_sorting_env")
world.add_task(task)
world.reset()

robot_name = task.get_params()["robot_name"]["value"]
robot = world.scene.get_object(robot_name)
arm = [robot.get_dof_index(name) for name in ARM_JOINTS]

# 손목 카메라 /rgb + /clock ROS2 발행 그래프 구성
camera_prim = find_wrist_camera_prim()
print(f"[ROS2] 손목 카메라 연결: {camera_prim} -> /rgb")
setup_ros2_camera_and_clock_graph(camera_prim)

# Lula IK 솔버 초기화 & 홈 자세(바닥을 향하는 자세)에서 아래 방향 쿼터니언 자동 계산
lula_ik = LulaKinematicsSolver(
    robot_description_path=str(ROBOT_DESC_YAML),
    urdf_path=str(ROBOT_URDF),
)
_, down_rot_mat = lula_ik.compute_forward_kinematics(EE_FRAME, np.radians(HOME_DEG))
down_quat = rot_matrix_to_quat(down_rot_mat)
ik_solver = ArticulationKinematicsSolver(robot, lula_ik, EE_FRAME)

# ROS2 /color_id 구독 노드 시작
rclpy.init()
ros_node = ColorIdSubscriber()

fsm = ColorPickPlaceFSM(
    robot=robot,
    task=task,
    arm_indices=arm,
    ik_solver=ik_solver,
    down_quat=down_quat,
    ros_node=ros_node,
)

# ============================================================
# 9. 메인 루프 (Stop -> Play 리셋 + ROS2 spin_once + FSM)
# ============================================================
reset_needed = False
try:
    while simulation_app.is_running():
        world.step(render=True)
        rclpy.spin_once(ros_node, timeout_sec=0.0)

        if world.is_stopped() and not reset_needed:
            reset_needed = True

        if world.is_playing():
            if reset_needed:
                world.reset()
                fsm.reset()
                reset_needed = False
                print("reset 완료: 처음부터 다시 시작")

            observations = world.get_observations()
            current_arm = observations[robot_name]["joint_positions"][arm]
            action = fsm.step(current_arm)
            if action is not None:
                robot.apply_action(action)
finally:
    ros_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    simulation_app.close()
