"""Isaac Sim(Run Remotely)에서 실행: Stage 내 0609 로봇과 카메라, ROS 2 카메라 토픽을 찾는다."""

import omni.usd
from pxr import UsdGeom

stage = omni.usd.get_context().get_stage()
if stage is None:
    print("열려 있는 USD Stage가 없습니다.")
else:
    robot_prims = [
        prim for prim in stage.Traverse()
        if "0609" in prim.GetName().lower() and prim.GetParent().GetPath() == "/World"
    ]
    if not robot_prims:
        robot_prims = [prim for prim in stage.Traverse() if "0609" in prim.GetName().lower()]

    if not robot_prims:
        print("Stage에서 이름에 '0609'가 포함된 로봇을 찾지 못했습니다.")
    else:
        for robot in robot_prims:
            robot_path = str(robot.GetPath())
            print(f"[로봇 발견] {robot_path}")

            cameras = [
                str(p.GetPath())
                for p in stage.Traverse()
                if str(p.GetPath()).startswith(robot_path) and p.IsA(UsdGeom.Camera)
            ]
            if cameras:
                print(f"  - 카메라 Prim: {cameras}")
            else:
                print("  - 하위에 Camera Prim이 없습니다.")

    # Stage 내 ROS 2 카메라 퍼블리셔 노드(OmniGraph)의 topicName 확인
    found_topics = []
    for prim in stage.Traverse():
        type_attr = prim.GetAttribute("node:type")
        topic_attr = prim.GetAttribute("inputs:topicName")
        if type_attr and topic_attr:
            node_type = str(type_attr.Get() or "")
            if "ROS2" in node_type or "Camera" in node_type:
                found_topics.append((str(prim.GetPath()), node_type, topic_attr.Get()))

    if found_topics:
        print("[ROS 2 카메라 관련 노드]")
        for path, node_type, topic in found_topics:
            print(f"  - {path} ({node_type}) -> topic: {topic}")
    else:
        print("Stage에 ROS 2 카메라 퍼블리셔(OmniGraph) 노드가 아직 없습니다.")
