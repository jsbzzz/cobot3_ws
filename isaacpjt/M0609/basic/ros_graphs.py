"""ROS2 OmniGraph 를 코드로 만드는 함수 모음 (L10_ros_graph.py 에서 단계별로 만든 것을 재사용용으로 정리)

사용법 (반드시 SimulationApp 생성 + isaacsim.ros2.bridge 확장을 켠 뒤, 씬을 연 다음에 import / 호출):

    import ros_graphs
    ros_graphs.reset_graph_root()            # 씬에 GUI 로 만든 그래프가 있으면 지운다
    ros_graphs.build_clock_graph()           # /clock
    ros_graphs.build_camera_graph()          # /rgb, /depth, camera_info
    ros_graphs.build_joint_state_graph(...)  # /joint_states (+ 선택: /joint_command 로 로봇 구동)
"""

import omni.graph.core as og
import omni.kit.app
from isaacsim.core.utils.prims import delete_prim, is_prim_path_valid

GRAPH_ROOT = "/World/Graph"
CAMERA_PRIM = "/World/m0609_rg2_d455/angle_bracket/realsense_d455/RSD455/Camera_OmniVision_OV9782_Color"
ART_ROOT = "/World/m0609_rg2_d455/root_joint"

keys = og.Controller.Keys


def reset_graph_root(root=GRAPH_ROOT):
    """기존 그래프 폴더를 지운다. 같은 경로에 그래프를 두 번 만들면 에러가 나기 때문"""
    if is_prim_path_valid(root):
        delete_prim(root)
        omni.kit.app.get_app().update()


def build_clock_graph(root=GRAPH_ROOT):
    """/clock 발행 (L10 2단계)"""
    og.Controller.edit(
        {"graph_path": f"{root}/ROS_Clock", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
            ],
            keys.SET_VALUES: [
                ("ReadSimTime.inputs:resetOnStop", False),
                ("PublishClock.inputs:topicName", "clock"),
            ],
        },
    )


def build_camera_graph(root=GRAPH_ROOT, camera_prim=CAMERA_PRIM, width=640, height=640,
                       frame_id="Camera", depth=True, camera_info=True):
    """카메라 이미지 발행 (L10 3단계). /rgb 는 항상, /depth 와 camera_info 는 선택"""
    publishers = [("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper", "rgb", "/rgb")]
    if depth:
        publishers.append(("DepthPublish", "isaacsim.ros2.bridge.ROS2CameraHelper", "depth", "/depth"))
    if camera_info:
        publishers.append(("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper", None, "camera_info"))

    create = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
        ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
    ]
    connect = [
        ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
        ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
    ]
    values = [
        ("RenderProduct.inputs:cameraPrim", camera_prim),
        ("RenderProduct.inputs:width", width),
        ("RenderProduct.inputs:height", height),
    ]
    # 발행 노드마다 같은 연결/설정을 반복하므로 for 문으로 만든다
    for name, node_type, img_type, topic in publishers:
        create.append((name, node_type))
        connect += [
            ("RenderProduct.outputs:execOut", f"{name}.inputs:execIn"),
            ("RenderProduct.outputs:renderProductPath", f"{name}.inputs:renderProductPath"),
            ("Context.outputs:context", f"{name}.inputs:context"),
        ]
        values += [
            (f"{name}.inputs:topicName", topic),
            (f"{name}.inputs:frameId", frame_id),
            (f"{name}.inputs:resetSimulationTimeOnStop", True),
        ]
        if img_type:   # CameraInfoHelper 에는 type 입력이 없다
            values.append((f"{name}.inputs:type", img_type))

    og.Controller.edit(
        {"graph_path": f"{root}/camera_graph", "evaluator_name": "execution"},
        {keys.CREATE_NODES: create, keys.CONNECT: connect, keys.SET_VALUES: values},
    )


def build_joint_state_graph(root=GRAPH_ROOT, art_root=ART_ROOT, subscribe_command=False):
    """/joint_states 발행 (L10 5단계).

    subscribe_command=True 이면 /joint_command 를 받아 ArticulationController 로 로봇을 움직인다.
    Python 코드(robot.apply_action)로 로봇을 직접 제어하는 스크립트에서는 False 로 둔다.
    (둘 다 켜면 같은 관절에 명령이 두 군데서 들어가 서로 덮어쓴다)
    """
    create = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
    ]
    connect = [
        ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
        ("Context.outputs:context", "PublishJointState.inputs:context"),
    ]
    values = [
        ("ReadSimTime.inputs:resetOnStop", False),
        ("PublishJointState.inputs:targetPrim", art_root),
        ("PublishJointState.inputs:topicName", "/joint_states"),
    ]
    if subscribe_command:
        create += [
            ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
        ]
        connect += [
            ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
            ("Context.outputs:context", "SubscribeJointState.inputs:context"),
            ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
            ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
        ]
        values += [
            ("SubscribeJointState.inputs:topicName", "/joint_command"),
            ("ArticulationController.inputs:targetPrim", art_root),
        ]

    og.Controller.edit(
        {"graph_path": f"{root}/ROS_JointStates", "evaluator_name": "execution"},
        {keys.CREATE_NODES: create, keys.CONNECT: connect, keys.SET_VALUES: values},
    )
