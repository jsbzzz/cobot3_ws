# Script Editor 전용: m0609 팔 관절 6개 각도를 실시간으로 보여주는 창
# 사용법: Window > Script Editor 에 붙여넣고 Run (Ctrl+Enter) -> Play(▶) 누르면 값이 갱신됨
# 다시 Run 해도 이전 창/구독은 자동으로 정리된다.

import builtins
import numpy as np
import omni.ui as ui
import omni.kit.app
import omni.timeline
from isaacsim.core.prims import SingleArticulation

ROBOT_PATH = "/World/m0609_rg2_d455"   # Stage 의 로봇 경로
ARM_JOINTS = [f"joint_{i}" for i in range(1, 7)]  # 팔 관절 이름 (없으면 앞 6개 사용)

# ---- 이전 실행 정리 ----
_old = getattr(builtins, "_joint_viewer", None)
if _old:
    _old["obj"].sub = None   # 업데이트 구독 해제
    _old["sub"] = None
    _old["window"].destroy()


class JointViewer:
    def __init__(self):
        self.robot = None
        self.indices = None
        self.timeline = omni.timeline.get_timeline_interface()

        self.window = ui.Window("M0609 Joint Angles", width=320, height=260)
        self.labels = []
        with self.window.frame:
            with ui.VStack(spacing=4):
                self.status = ui.Label("Play(▶)를 누르세요", height=20)
                for i in range(6):
                    with ui.HStack(height=24):
                        ui.Label(f"J{i + 1}", width=40)
                        self.labels.append(ui.Label("-", alignment=ui.Alignment.RIGHT_CENTER))
                        ui.Label(" deg", width=40)

        self.sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="joint_viewer")
        )

    def _init_robot(self):
        self.robot = SingleArticulation(prim_path=ROBOT_PATH)
        self.robot.initialize()
        names = list(self.robot.dof_names)
        if all(n in names for n in ARM_JOINTS):
            self.indices = [names.index(n) for n in ARM_JOINTS]
        else:
            self.indices = list(range(6))
        self.status.text = "관절: " + ", ".join(names[i] for i in self.indices)

    def _on_update(self, _e):
        if not self.timeline.is_playing():
            self.robot = None  # Stop 하면 다시 초기화하도록
            self.status.text = "Play(▶)를 누르세요"
            return
        try:
            if self.robot is None:
                self._init_robot()
            q = self.robot.get_joint_positions()
            if q is None:
                return
            deg = np.degrees(np.asarray(q)[self.indices])
            for lbl, d in zip(self.labels, deg):
                lbl.text = f"{d:8.2f}"
        except Exception as ex:
            self.robot = None
            self.status.text = f"초기화 실패: {ex}"


_v = JointViewer()
builtins._joint_viewer = {"window": _v.window, "sub": _v.sub, "obj": _v}
