"""[L3 예제] 원뿔 추가 및 위치, 자세 복사

실행: isaac_python L3_scene_objects.py
"""

# ============================================================
# 1. 앱 시작
# ============================================================
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# ============================================================
# 2. Isaac 모듈 import
# ============================================================
import numpy as np
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCone, VisualCone
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# 조명 설정
omni.kit.commands.execute(
    "SetLightingMenuModeCommand", lighting_mode="Default"
)

# 1) 빨간 원뿔: 중력 적용, X축으로 60도 기울임
red = world.scene.add(
    DynamicCone(
        prim_path="/World/Red",
        name="red",
        position=np.array([0.4, 0.0, 0.5]),
        orientation=euler_angles_to_quat(
            np.array([60.0, 0.0, 0.0]), degrees=True
        ),
        radius=0.05,
        height=0.15,
        color=np.array([1.0, 0.0, 0.0]),
    )
)

# 2) 노란 원뿔: 중력 미적용
yellow = world.scene.add(
    VisualCone(
        prim_path="/World/Yellow",
        name="yellow",
        position=np.array([0.4, 0.0, 0.8]),
        radius=0.05,
        height=0.15,
        color=np.array([1.0, 1.0, 0.0]),
    )
)

world.reset()

# ============================================================
# 3. 메인 루프
# ============================================================
frame = 0

while simulation_app.is_running():
    world.step(render=True)
    frame += 1

    # 빨간 원뿔의 위치와 자세 조회
    red_position, red_orientation = red.get_world_pose()

    # 노란 원뿔이 빨간 원뿔보다 0.3m 위에서
    # 동일한 위치 변화와 자세를 따라가도록 설정
    yellow.set_world_pose(
        position=red_position + np.array([0.0, 0.0, 0.3]),
        orientation=red_orientation
    )

    # 60프레임마다 빨간 원뿔의 위치와 각도 출력
    if frame % 60 == 0:
        red_angles = quat_to_euler_angles(
            red_orientation, degrees=True
        )

        print(f"{frame} 프레임 | red 위치 {np.round(red_position, 3)} | 각도(deg) {np.round(red_angles, 1)}")

# ============================================================
# 4. 종료
# ============================================================
simulation_app.close()