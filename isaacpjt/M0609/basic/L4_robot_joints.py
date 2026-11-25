"""[L4 예제] M0609 로봇 불러오기와 관절 제어

실행: isaac_python L4_robot_joints.py
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
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction


ROBOT_USD = Path(__file__).resolve().parents[1] / "robot" / "m0609_rg2_d455.usd"  # isaacpjt/M0609/robot

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
# 조명: 화면 위 Lighting 메뉴에서 "Default" 를 고른 것과 같다
omni.kit.commands.execute("SetLightingMenuModeCommand", lighting_mode="Default")

# (가) 로봇 USD 참조: 파일의 기본 prim(로봇)을 /World/m0609_rg2_d455 에 넣는다
add_reference_to_stage(usd_path=str(ROBOT_USD), prim_path="/World/m0609_rg2_d455")
robot = world.scene.add(SingleArticulation(prim_path="/World/m0609_rg2_d455", name="m0609"))
world.reset()  # reset 후에 관절 정보를 읽을 수 있다

# (나) 관절 정보
print("관절 개수:", robot.num_dof)
for index, name in enumerate(robot.dof_names):
    print(f"  {index:2d} {name}")
arm = [robot.get_dof_index(f"joint_{i}") for i in range(1, 7)]
finger = robot.get_dof_index("finger_joint")
print("arm (팔 관절 6개가 관절 배열의 몇 번째인지):", arm)
print("finger (그리퍼가 관절 배열의 몇 번째인지):", finger)

# (다) 순간이동: joint_1 45 deg, 그리퍼 열기
start = np.zeros(robot.num_dof)
start[arm] = np.radians([45, 0, 90, 0, 90, 0])  # 팔
start[finger] = 0.0                             # 그리퍼 열기 (0.0 rad)
print("순간이동 명령 start (관절 12개, rad):", np.round(start, 3).tolist())
robot.set_joint_positions(start)
q = robot.get_joint_positions()
print(f"순간이동 후 | joint_1 {np.degrees(q[arm[0]]):6.1f} deg | 그리퍼 {abs(q[finger]):.2f} rad")


# ============================================================
# 3. 메인 루프 (창이 열려 있는 동안 한 프레임씩 진행)
# ============================================================
MOVE_FRAME = 100  # 첫 이동을 시작할 프레임
STEP_DEG = 1.5     # 한 프레임에 바꿀 수 있는 모터 목표 각도
goal_deg = -45   # 이번 이동의 최종 목표 각도
target = start.copy()
frame = 0

while simulation_app.is_running():
    world.step(render=True)
    frame += 1

    if frame < MOVE_FRAME:
        continue

    cur_deg = np.degrees(robot.get_joint_positions()[arm[0]])
    if frame == MOVE_FRAME:
        # 첫 출발 때 실제 관절 각도를 모터 명령의 시작점으로 잡는다.
        command_deg = cur_deg
    elif command_deg == goal_deg and abs(cur_deg - goal_deg) < 1.0:
        # 최종 목표를 이미 보냈고 실제 관절도 1도 미만으로 가까워지면 방향을 바꾼다.
        print(f"{frame} 프레임 | joint_1 {cur_deg:6.1f} deg, {goal_deg} deg 도착")
        goal_deg = -goal_deg

    # 남은 각도가 한 번에 움직일 수 있는 범위면 최종 목표를 바로 보낸다.
    # 그보다 멀면 목표 방향으로 STEP_DEG만큼 모터 명령을 바꾼다.
    remaining_deg = goal_deg - command_deg
    if abs(remaining_deg) <= STEP_DEG:
        command_deg = goal_deg
    else:
        command_deg += np.copysign(STEP_DEG, remaining_deg)

    target[arm[0]] = np.radians(command_deg)
    target[finger] = 0.9 if goal_deg < 0 else 0.0  # 왼쪽 닫기, 오른쪽 열기
    robot.apply_action(ArticulationAction(joint_positions=target))
# ============================================================
# 4. 종료
# ============================================================
simulation_app.close()
