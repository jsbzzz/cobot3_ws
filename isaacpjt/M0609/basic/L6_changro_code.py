

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
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

ROBOT_USD = Path(__file__).resolve().parents[1] / "robot" / "m0609_rg2_d455.usd"
ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
HOME_DEG = [0, 0, 90, 0, 90, 0]
CUBE_POSITION = np.array([0.45, 0.2, 0.026])
CUBE_SIZE = 0.0515


# ------------------------------------------------------------
# (가) Task: 장면(바닥 + M0609 + 파란 큐브), 시작 자세, 관측값
# ------------------------------------------------------------
class LookAtCubeTask(BaseTask):
    """바닥, M0609, 파란 큐브를 만들고 로봇 관절과 큐브 위치를 관측값으로 돌려준다"""

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()
        add_reference_to_stage(usd_path=str(ROBOT_USD), prim_path="/World/m0609_rg2_d455")
        self._robot = scene.add(SingleArticulation(prim_path="/World/m0609_rg2_d455", name="m0609"))
        self._cube = scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=CUBE_POSITION,
                size=CUBE_SIZE,
                color=np.array([0.0, 0.0, 1.0]),
            )
        )
        self._task_objects[self._robot.name] = self._robot
        self._task_objects[self._cube.name] = self._cube

    def post_reset(self):
        home = np.zeros(self._robot.num_dof)
        for name, angle in zip(ARM_JOINTS, np.radians(HOME_DEG)):
            home[self._robot.get_dof_index(name)] = angle
        self._robot.set_joint_positions(home)

    def get_observations(self):
        cube_position, _ = self._cube.get_world_pose()
        return {
            self._robot.name: {"joint_positions": self._robot.get_joint_positions()},
            self._cube.name: {"position": cube_position},
        }

    def get_params(self):
        return {
            "robot_name": {"value": self._robot.name},
            "cube_name": {"value": self._cube.name},
        }


# ------------------------------------------------------------
# (나) Controller: 큐브 위치를 받아 joint_1 을 큐브 방향으로 돌린다
# ------------------------------------------------------------
class LookAtCubeController(BaseController):
    """원점(로봇)에서 본 큐브 방향 각도로 joint_1 만 회전시킨다 (나머지 관절은 홈 자세)"""

    def __init__(self, name, arm_indices):
        super().__init__(name)
        self._arm_indices = arm_indices

    def reset(self):
        pass

    def forward(self, cube_position):
        # 원점(로봇)에서 큐브 (x, y)를 바라보는 각도 (rad)
        angle = np.arctan2(cube_position[1], cube_position[0])
        goal_arm = np.radians(HOME_DEG)
        goal_arm[0] = angle
        return ArticulationAction(joint_positions=goal_arm, joint_indices=self._arm_indices)


# ------------------------------------------------------------
# (다) 등록과 준비
# ------------------------------------------------------------
world = World(stage_units_in_meters=1.0)
omni.kit.commands.execute("SetLightingMenuModeCommand", lighting_mode="Default")
task = LookAtCubeTask(name="look_at_cube")
world.add_task(task)
world.reset()

params = task.get_params()
robot_name = params["robot_name"]["value"]
cube_name = params["cube_name"]["value"]
robot = world.scene.get_object(robot_name)
arm = [robot.get_dof_index(name) for name in ARM_JOINTS]
controller = LookAtCubeController("look_at_cube_controller", arm_indices=arm)

# ============================================================
# 3. 메인 루프 (예제 구조 유지 + 60프레임마다 출력)
# ============================================================
reset_needed = False
frame = 0
while simulation_app.is_running():
    world.step(render=True)

    # 1) Stop 감지
    if world.is_stopped() and not reset_needed:
        reset_needed = True
    if world.is_playing():
        # 2) Stop 다음 첫 Play 프레임: 장면, 컨트롤러, 프레임 카운터 리셋
        if reset_needed:
            world.reset()
            controller.reset()
            frame = 0
            reset_needed = False
            print("reset 완료: 처음부터 다시 시작")

        frame += 1
        # 3) 관측 (Task)
        observations = world.get_observations()
        current_arm = observations[robot_name]["joint_positions"][arm]
        cube_pos = observations[cube_name]["position"]
        # 4) 계산 (Controller)
        action = controller.forward(cube_pos)
        # 5) 명령
        robot.apply_action(action)

        if frame % 60 == 0:
            print(
                f"{frame} 프레임 | 큐브 x {cube_pos[0]:5.2f} y {cube_pos[1]:5.2f} m"
                f" | joint_1 {np.degrees(current_arm[0]):6.1f} deg"
            )

# ============================================================
# 4. 종료
# ============================================================
simulation_app.close()
