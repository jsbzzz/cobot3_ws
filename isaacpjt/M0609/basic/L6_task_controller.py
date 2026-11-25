"""[L6 연습 문제] 큐브를 바라보는 로봇 만들기

실행: isaac_python L6_task_controller.py
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
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.objects import DynamicCuboid  # 추가

ROBOT_USD = Path(__file__).resolve().parents[1] / "robot" / "m0609_rg2_d455.usd"
ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
HOME_DEG = [0, 0, 90, 0, 90, 0]


# ------------------------------------------------------------
# (가) Task: 로봇이 놓인 환경 (장면 + 시작 자세 + 관측값)
# ------------------------------------------------------------
class RobotEnvTask(BaseTask):
    """장면을 만들고, 시작 자세를 맞추고, 관측값을 돌려준다"""

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        add_reference_to_stage(
            usd_path=str(ROBOT_USD),
            prim_path="/World/m0609_rg2_d455"
        )

        self._robot = scene.add(SingleArticulation(
            prim_path="/World/m0609_rg2_d455",
            name="m0609"
        ))

        self._task_objects[self._robot.name] = self._robot

        # 추가: 파란색 큐브 생성
        self._cube = scene.add(DynamicCuboid(
            prim_path="/World/cube",
            name="cube",
            position=np.array([0.45, 0.2, 0.026]),
            size=0.0515,
            color=np.array([0.0, 0.0, 1.0])
        ))

        # 추가: Task에 큐브 등록
        self._task_objects[self._cube.name] = self._cube

    # 언제: 매 world.reset() 끝에
    # 할 일: 시작 자세 맞추기
    def post_reset(self):
        home = np.zeros(self._robot.num_dof)

        for name, angle in zip(ARM_JOINTS, np.radians(HOME_DEG)):
            home[self._robot.get_dof_index(name)] = angle

        self._robot.set_joint_positions(home)

    # 언제: world.get_observations() 를 부를 때
    def get_observations(self):
        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions()
            },

            # 추가: 큐브 위치 관측
            self._cube.name: {
                "position": self._cube.get_world_pose()[0]
            }
        }

    def get_params(self):
        return {"robot_name": {"value": self._robot.name}}


# ------------------------------------------------------------
# (나) Controller: 관측값을 받아 명령을 계산한다
# ------------------------------------------------------------
class SwingController(BaseController):
    """joint_1을 큐브 방향으로 회전시킨다"""

    def __init__(self, name, arm_indices):
        super().__init__(name)
        self._arm_indices = arm_indices
        self.reset()

    def reset(self):
        self.goal_deg = 0.0

    # 수정: 현재 관절값 대신 큐브 위치를 받는다
    def forward(self, cube_position):

        # 1) 추가: 큐브의 x, y 좌표
        x = cube_position[0]
        y = cube_position[1]

        # 2) 수정: 큐브 방향의 각도 계산
        self.goal_deg = np.degrees(np.arctan2(y, x))

        # 3) 목표 자세: 홈 자세에서 joint_1만 변경
        goal_arm = np.radians(HOME_DEG)
        goal_arm[0] = np.radians(self.goal_deg)

        # 4) 명령 (팔 관절 6개에만)
        return ArticulationAction(
            joint_positions=goal_arm,
            joint_indices=self._arm_indices
        )


# ------------------------------------------------------------
# (다) 등록과 준비
# ------------------------------------------------------------
world = World(stage_units_in_meters=1.0)

omni.kit.commands.execute(
    "SetLightingMenuModeCommand",
    lighting_mode="Default"
)

task = RobotEnvTask(name="robot_env")
world.add_task(task)
world.reset()

robot_name = task.get_params()["robot_name"]["value"]
robot = world.scene.get_object(robot_name)

arm = [robot.get_dof_index(name) for name in ARM_JOINTS]

controller = SwingController(
    "swing_controller",
    arm_indices=arm
)

# ============================================================
# 3. 메인 루프 (창이 열려 있는 동안 한 프레임씩 진행)
# ============================================================
reset_needed = False

while simulation_app.is_running():
    world.step(render=True)

    # 1) Stop 을 감지해 둔다
    if world.is_stopped() and not reset_needed:
        reset_needed = True

    if world.is_playing():

        # 2) Stop 다음 첫 Play 프레임
        if reset_needed:
            world.reset()
            controller.reset()
            reset_needed = False
            print("reset 완료: 처음부터 다시 시작")

        # 3) 관측 (Task)
        observations = world.get_observations()
        current_arm = observations[robot_name]["joint_positions"][arm]

        # 추가: 큐브 위치 가져오기
        cube_position = observations["cube"]["position"]

        # 4) 계산 (Controller)
        # 수정: 큐브 위치 전달
        action = controller.forward(cube_position)

        # 5) 명령
        robot.apply_action(action)

        # 수정: 60프레임마다 출력
        if world.current_time_step_index % 60 == 0:
            print(
                f"{world.current_time_step_index} 프레임 | "
                f"큐브 x {cube_position[0]:5.2f} "
                f"y {cube_position[1]:5.2f} m | "
                f"joint_1 {np.degrees(current_arm[0]):6.1f} deg"
            )

# ============================================================
# 4. 종료
# ============================================================
simulation_app.close()