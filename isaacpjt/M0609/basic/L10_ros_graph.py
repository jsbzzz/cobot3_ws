# L10: GUI 로 만들었던 ROS2 연결(scenes/pick_place_cube_ros.usd)을 코드로 다시 만들기
#
# 단계
#   1. 뼈대: SimulationApp -> ROS2 bridge 확장 켜기 -> 원본 씬 열기      (완료)
#   2. Clock 그래프: /clock 토픽 발행                                    (완료)
#   3. 카메라 그래프: /rgb, /depth, camera_info 발행                     (완료)
#   4. 실행 루프 정리                                                    (완료)
#   5. 관절 그래프: /joint_states 발행 + /joint_command 받아서 로봇 구동  (완료)
#
# 실행 방법 (Isaac Sim 을 띄우는 터미널)
#   isaac_ros                           # ~/.bashrc 함수: 브리지 내장 jazzy 라이브러리를 LD_LIBRARY_PATH 에 추가
#   ~/isaacsim/python.sh L10_ros_graph.py
#
#   주의: 이 터미널에서는 source /opt/ros/jazzy/setup.bash (ros_set) 를 하지 않는다.
#         시스템 jazzy 는 Python 3.12, Isaac Sim 5.1 은 Python 3.11 이라 섞이면 rclpy 로딩이 실패한다.
#         ROS_DISTRO=jazzy, RMW_IMPLEMENTATION=rmw_fastrtps_cpp 는 ~/.bashrc 에 이미 있음.
#
# 토픽 확인 (다른 터미널): ros_set 후 ros2 topic list

# ---------------------------------------------------------------
# 1단계: 뼈대
# ---------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})   # 항상 맨 먼저. 이 뒤에야 omni / isaacsim 모듈 import 가능

from isaacsim.core.utils import extensions, stage

extensions.enable_extension("isaacsim.ros2.bridge")   # GUI 의 Extensions 창에서 켜는 것과 같음
simulation_app.update()                                # 확장이 실제로 로드되도록 한 프레임 진행

import omni.graph.core as og                           # OmniGraph(Action Graph) 를 코드로 다루는 모듈
from isaacsim.core.api import SimulationContext

SCENE = "/home/rokey/cobot3_ws/isaacpjt/M0609/scenes/pick_place_cube.usd"  # ROS 그래프가 "없는" 원본 씬
stage.open_stage(SCENE)
while stage.is_stage_loading():   # 로봇 USD 같은 참조 파일까지 다 읽힐 때까지 대기
    simulation_app.update()        # (안 기다리면 아래에서 카메라 prim 을 못 찾을 수 있음)

GRAPH_ROOT = "/World/Graph"   # 팀원과 만든 씬과 같은 위치에 그래프를 둔다

# 원본 씬에도 GUI 로 만든 ROS_Clock 그래프가 이미 들어 있다.
# 같은 경로에 그래프를 또 만들면 "A graph already exists at this path" 에러가 나므로,
# 기존 그래프 폴더를 통째로 지우고 아래에서 전부 코드로 새로 만든다.
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid

if is_prim_path_valid(GRAPH_ROOT):
    delete_prim(GRAPH_ROOT)
    simulation_app.update()

# ---------------------------------------------------------------
# 2단계: Clock 그래프  (GUI: Tools > Robotics > ROS 2 OmniGraphs > Clock)
#
#   OnPlaybackTick ──tick──────────────▶ PublishClock.execIn      (매 프레임 실행 신호)
#   ReadSimTime ──simulationTime───────▶ PublishClock.timeStamp   (보낼 시간 값)
#   Context ──context──────────────────▶ PublishClock.context     (ROS2 연결 정보, domain id)
# ---------------------------------------------------------------
keys = og.Controller.Keys

og.Controller.edit(
    {
        "graph_path": f"{GRAPH_ROOT}/ROS_Clock",   # Stage 에 생길 그래프 prim 경로
        "evaluator_name": "execution",             # Action Graph (실행 신호 tick 으로 동작하는 그래프)
    },
    {
        # (노드 이름, 노드 타입): GUI 에서 노드를 끌어다 놓는 것
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),        # Play 중 매 프레임 tick 발생
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),  # 현재 시뮬레이션 시간(초)
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),               # ROS2 노드/도메인 설정
            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),     # rosgraph_msgs/Clock 발행
        ],
        # ("보내는 노드.출력", "받는 노드.입력"): GUI 에서 선을 잇는 것
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ("Context.outputs:context", "PublishClock.inputs:context"),
        ],
        # ("노드.입력", 값): GUI 의 Property 창에서 값을 바꾸는 것
        keys.SET_VALUES: [
            ("ReadSimTime.inputs:resetOnStop", False),   # Stop 해도 시간을 0 으로 되돌리지 않음 (씬 파일 값과 동일)
            ("PublishClock.inputs:topicName", "clk"),  # 토픽 이름. 기본값이 clock 이지만 명시
        ],
    },
)

# ---------------------------------------------------------------
# 3단계: 카메라 그래프  (GUI: Tools > Robotics > ROS 2 OmniGraphs > Camera)
#
#   OnPlaybackTick ─tick─▶ RunOnce ─step─▶ RenderProduct ─execOut──────────┬─▶ RGBPublish.execIn
#                         (Play 후 1번만)   (카메라 -> 렌더 결과 만들기)    ├─▶ DepthPublish.execIn
#                                                                           └─▶ CameraInfoPublish.execIn
#                                           RenderProduct ─renderProductPath─▶ 위 3개 노드 .renderProductPath
#   Context ─context─────────────────────────────────────────────────────────▶ 위 3개 노드 .context
#
#   RunOnce 가 왜 필요?: 발행 노드(Helper)는 처음 1번 실행될 때 "매 프레임 발행"을 스스로 등록한다.
#   그래서 RenderProduct 생성 + Helper 등록은 Play 직후 1번이면 충분하고, 매 프레임 다시 만들 필요가 없다.
# ---------------------------------------------------------------
# 로봇 USD 안의 RealSense D455 컬러 카메라 (씬 파일의 카메라 경로와 동일)
CAMERA_PRIM = "/World/m0609_rg2_d455/angle_bracket/realsense_d455/RSD455/Camera_OmniVision_OV9782_Color"
IMG_W, IMG_H = 640, 640     # 발행할 이미지 해상도
FRAME_ID = "Camera"         # 메시지 header.frame_id (TF 에서 이 카메라를 부르는 이름)

og.Controller.edit(
    {"graph_path": f"{GRAPH_ROOT}/camera_graph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),     # 첫 프레임에만 실행 신호 통과
            ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),    # 카메라 -> 렌더 결과(RenderProduct)
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),             # sensor_msgs/Image (컬러)
            ("DepthPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),           # sensor_msgs/Image (깊이) - 같은 노드 타입, type 만 다름
            ("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),  # sensor_msgs/CameraInfo (내부 파라미터)
        ],
        keys.CONNECT: [
            # 실행 신호 흐름
            ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
            ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
            ("RenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
            ("RenderProduct.outputs:execOut", "DepthPublish.inputs:execIn"),
            ("RenderProduct.outputs:execOut", "CameraInfoPublish.inputs:execIn"),
            # 데이터: 어떤 렌더 결과를 발행할지
            ("RenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
            ("RenderProduct.outputs:renderProductPath", "DepthPublish.inputs:renderProductPath"),
            ("RenderProduct.outputs:renderProductPath", "CameraInfoPublish.inputs:renderProductPath"),
            # 데이터: ROS2 연결 정보
            ("Context.outputs:context", "RGBPublish.inputs:context"),
            ("Context.outputs:context", "DepthPublish.inputs:context"),
            ("Context.outputs:context", "CameraInfoPublish.inputs:context"),
        ],
        keys.SET_VALUES: [
            # RenderProduct: 어떤 카메라로, 몇 해상도로 렌더할지
            ("RenderProduct.inputs:cameraPrim", CAMERA_PRIM),
            ("RenderProduct.inputs:width", IMG_W),
            ("RenderProduct.inputs:height", IMG_H),
            # RGB
            ("RGBPublish.inputs:type", "rgb"),
            ("RGBPublish.inputs:topicName", "/rgb"),
            ("RGBPublish.inputs:frameId", FRAME_ID),
            ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
            # Depth
            ("DepthPublish.inputs:type", "depth"),
            ("DepthPublish.inputs:topicName", "/depth"),
            ("DepthPublish.inputs:frameId", FRAME_ID),
            ("DepthPublish.inputs:resetSimulationTimeOnStop", True),
            # CameraInfo
            ("CameraInfoPublish.inputs:topicName", "camera_info"),
            ("CameraInfoPublish.inputs:frameId", FRAME_ID),
            ("CameraInfoPublish.inputs:resetSimulationTimeOnStop", True),
        ],
    },
)

# ---------------------------------------------------------------
# 5단계: 관절 그래프  (GUI: Tools > Robotics > ROS 2 OmniGraphs > Joint States)
#
#   [발행] Isaac -> ROS : 현재 관절 각도를 /joint_states 로 내보냄
#     OnPlaybackTick ─tick──────────▶ PublishJointState.execIn
#     ReadSimTime ─simulationTime───▶ PublishJointState.timeStamp
#
#   [구독] ROS -> Isaac : /joint_command 로 받은 목표 각도로 로봇을 움직임
#     OnPlaybackTick ─tick──────────▶ SubscribeJointState.execIn
#     OnPlaybackTick ─tick──────────▶ ArticulationController.execIn
#     SubscribeJointState ─jointNames / positionCommand / velocityCommand / effortCommand─▶ ArticulationController
#
#   Context ─context─▶ PublishJointState, SubscribeJointState
#   (ArticulationController 는 Isaac 내부 노드라 ROS Context 가 필요 없음)
# ---------------------------------------------------------------
# PhysicsArticulationRootAPI 가 붙은 prim (로봇 USD 안의 root_joint). GUI 메뉴의 "Articulation Root" 칸에 넣던 값
ART_ROOT = "/World/m0609_rg2_d455/root_joint"

og.Controller.edit(
    {"graph_path": f"{GRAPH_ROOT}/ROS_JointStates", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),        # sensor_msgs/JointState 발행
            ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),    # sensor_msgs/JointState 구독
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),  # 받은 명령을 관절 드라이브에 적용
        ],
        keys.CONNECT: [
            # 발행
            ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
            ("Context.outputs:context", "PublishJointState.inputs:context"),
            # 구독
            ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
            ("Context.outputs:context", "SubscribeJointState.inputs:context"),
            # 구독 결과 -> 로봇 구동
            ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
            ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
        ],
        keys.SET_VALUES: [
            ("ReadSimTime.inputs:resetOnStop", False),
            ("PublishJointState.inputs:targetPrim", ART_ROOT),       # 어떤 로봇의 관절을 읽을지
            ("PublishJointState.inputs:topicName", "/joint_states"),
            ("SubscribeJointState.inputs:topicName", "/joint_command"),
            ("ArticulationController.inputs:targetPrim", ART_ROOT),  # 어떤 로봇을 움직일지
        ],
    },
)

# ---------------------------------------------------------------
# 4단계: 실행 루프
#
#   SimulationContext = 물리 시간을 관리하는 객체 (World 의 부모 클래스. 로봇/Task 기능만 뺀 것)
#   step(render=True) 1번 = 물리 1스텝 + 렌더 1번 + OmniGraph 실행 1번
#   -> 매 step 마다 OnPlaybackTick 이 tick 을 내고, /clock 과 카메라 토픽이 1번씩 발행된다
# ---------------------------------------------------------------
simulation_app.update()   # 방금 만든 그래프를 Stage 에 반영

simulation_context = SimulationContext(stage_units_in_meters=1.0)   # 씬 단위: m (open_stage 이후에 만들어야 함)
simulation_context.initialize_physics()   # 물리 엔진 준비 (관절/강체 핸들 생성)
simulation_context.play()                 # Play(▶). 여기서부터 OnPlaybackTick 이 tick 을 낸다

try:
    while simulation_app.is_running():    # 창을 닫으면 False
        simulation_context.step(render=True)

        # 여기에 매 프레임 할 일을 넣는다 (예: 로봇 제어, ROS 에서 받은 명령 처리)

except KeyboardInterrupt:                 # 터미널에서 Ctrl+C 로 멈춘 경우
    pass
finally:
    simulation_context.stop()             # Stop(■). 발행 노드들도 정리됨
    simulation_app.close()                # Isaac Sim 종료
