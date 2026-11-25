"""로컬 PC(ROS 2)에서 실행: 카메라 이미지 토픽을 찾아 echo 출력한다."""

import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def find_image_topics(node: Node, wait_sec: float = 3.0) -> list[str]:
    """ROS 2 그래프에서 sensor_msgs/msg/Image 토픽을 찾는다."""
    deadline = time.monotonic() + wait_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        topics = [
            name
            for name, types in node.get_topic_names_and_types()
            if "sensor_msgs/msg/Image" in types
        ]
        if topics:
            return topics
    return []


class CameraTopicEcho(Node):
    """지정한 Image 토픽을 구독해 헤더·해상도·인코딩 정보를 출력한다."""

    def __init__(self, topic_name: str):
        super().__init__("camera_topic_echo")
        self.create_subscription(
            Image,
            topic_name,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"구독 시작 (echo): {topic_name}")

    def _on_image(self, msg: Image) -> None:
        stamp = f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
        sample = list(msg.data[:8])
        print(
            f"---\n"
            f"header:\n"
            f"  stamp: {stamp}\n"
            f"  frame_id: '{msg.header.frame_id}'\n"
            f"height: {msg.height}, width: {msg.width}\n"
            f"encoding: '{msg.encoding}', step: {msg.step}\n"
            f"data[:8]: {sample} (total {len(msg.data)} bytes)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Sim 카메라 ROS 2 토픽 echo")
    parser.add_argument("--topic", help="직접 지정할 Image 토픽 이름 (생략 시 자동 탐색)")
    args = parser.parse_args()

    rclpy.init()
    probe = rclpy.create_node("camera_topic_probe")
    try:
        topic = args.topic
        if not topic:
            topics = find_image_topics(probe, wait_sec=3.0)
            if not topics:
                print("sensor_msgs/msg/Image 토픽을 찾지 못했습니다.")
                print("원격 Isaac Sim에서 Play 상태, ROS2 Bridge, 카메라 퍼블리셔와 ROS_DOMAIN_ID를 확인하세요.")
                return
            print(f"발견된 Image 토픽: {topics}")
            topic = topics[0]
    finally:
        probe.destroy_node()

    echo_node = CameraTopicEcho(topic)
    try:
        rclpy.spin(echo_node)
    except KeyboardInterrupt:
        pass
    finally:
        echo_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
