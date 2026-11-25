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
import time 
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
MOVE_FRAME = 100

target_deg = -45  # 추가: 목표 각도
tolerance = 1.0   # 추가: 도착 허용 오차

frame = 0
while simulation_app.is_running():
    world.step(render=True)
    time.sleep(0.03)
    frame += 1

    if frame == MOVE_FRAME:
        target = start.copy()
        target[arm] = np.radians([-45, 0, 90, 0, 90, 0])
        target[finger] = 0.9

        print("모터 목표 명령 target (관절 12개, rad):",
              np.round(target, 3).tolist())

        robot.apply_action(ArticulationAction(joint_positions=target))

    # 추가: 현재 관절 각도 확인
    if frame >= MOVE_FRAME:
        q = robot.get_joint_positions()
        current_deg = np.degrees(q[arm[0]])

        # 추가: 목표 각도에 도착했는지 확인
        if abs(current_deg - target_deg) <= tolerance:

            print(f"{frame} 프레임 | joint_1 {current_deg:6.1f} deg, {target_deg} deg 도착")

            # 추가: 반대 방향으로 목표 변경
            target_deg = -target_deg

            target = start.copy()
            target[arm[0]] = np.radians(target_deg)

            # 추가: 그리퍼 열고 닫기
            if target_deg == -45:
                target[finger] = 0.9
            else:
                target[finger] = 0.0

            robot.apply_action(ArticulationAction(joint_positions=target))

# ============================================================
# 4. 종료
# ============================================================
simulation_app.close()