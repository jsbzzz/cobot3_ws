from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)                # 2. World ì”¬ì˜ ë‹¨ìœ„ë¥¼ ë¯¸í„°ë¡œ ì„¤ì •í•œë‹¤, ì´í›„ ëª¨ë“  ì¢Œí‘œê°’ì´ ì‹¤ì œ ë¯¸í„° ë‹¨ìœ„ë¡œ ë™ìž‘í•œë‹¤.
stage = omni.usd.get_context().get_stage()              # 3. Stage USD ì”¬ ë°ì´í„°ì— ì§ì ‘ ì ‘ê·¼í•˜ëŠ” ì§„ìž…ì ì´ë‹¤

cube_prim = DynamicCuboid(                              # 4. Prim
    prim_path="/World/BlueCube",
    name="blue_cube",
    position=np.array([0.0, 0.0, 0.5]),
    scale=np.array([0.15, 0.15, 0.15]),
    color=np.array([0.0, 0.0, 1.0]),
)

world.scene.add_default_ground_plane() #ë°”ë‹¥             # 5. Scene
world.scene.add(cube_prim) #ì›ë“œì— ì˜¤ë¸Œì íŠ¸ ë“±ë¡

world.reset() #ë¬¼ë¦¬ë¥¼ ì´ˆê¸°í™”í•˜ê³  ì”¬ì„ í™•ì •

while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)

simulation_app.close()