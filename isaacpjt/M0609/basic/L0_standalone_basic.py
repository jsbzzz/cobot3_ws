from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)                # 2. World 씬의 단위를 미터로 설정한다, 이후 모든 좌표값이 실제 미터 단위로 동작한다.
stage = omni.usd.get_context().get_stage()              # 3. Stage USD 씬 데이터에 직접 접근하는 진입점이다

cube_prim = DynamicCuboid(                              # 4. Prim
    prim_path="/World/BlueCube",
    name="blue_cube",
    position=np.array([0.0, 0.0, 0.5]),
    scale=np.array([0.15, 0.15, 0.15]),
    color=np.array([0.0, 0.0, 1.0]),
)

world.scene.add_default_ground_plane() #바닥             # 5. Scene
world.scene.add(cube_prim) #원드에 오브젝트 등록

world.reset() #물리를 초기화하고 씬을 확정

while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)

simulation_app.close()