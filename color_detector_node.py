"""/rgb 토픽에서 중앙 영역 큐브 색상을 판별해 /color_id(std_msgs/Int32)를 1회 발행하는 ROS 2 노드.

- 파랑(Blue) -> 1
- 초록(Green) -> 2
- 빨강(Red) -> 3
- 판별 완료 시 /color_id를 1회만 발행하며, 큐브가 화면에서 사라진 뒤 다시 감지되면 새로 발행한다.
"""

from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

COLOR_BLUE = 1
COLOR_GREEN = 2
COLOR_RED = 3

COLOR_LABELS = {
    COLOR_BLUE: "파랑(1)",
    COLOR_GREEN: "초록(2)",
    COLOR_RED: "빨강(3)",
}

# OpenCV HSV 범위 (H: 0~179, S: 0~255, V: 0~255)
COLOR_HSV_RANGES = {
    COLOR_BLUE: [(np.array([95, 60, 50]), np.array([130, 255, 255]))],
    COLOR_GREEN: [(np.array([40, 60, 50]), np.array([85, 255, 255]))],
    # 빨강은 Hue가 0과 180 양끝에 걸쳐 있어 두 구간을 합산
    COLOR_RED: [
        (np.array([0, 60, 50]), np.array([10, 255, 255])),
        (np.array([170, 60, 50]), np.array([179, 255, 255])),
    ],
}


def image_msg_to_bgr(msg: Image) -> np.ndarray | None:
    """sensor_msgs/Image를 BGR numpy 배열로 변환한다."""
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        channels = 3
    elif enc in ("rgba8", "bgra8"):
        channels = 4
    else:
        return None

    img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, channels))
    if enc.startswith("rgb"):
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
    return img[:, :, :3]


def detect_center_cube_color(bgr_img: np.ndarray, roi_ratio: float = 0.4, min_ratio: float = 0.05) -> int | None:
    """이미지 중앙 ROI 영역에서 일정 비율 이상 차지하는 파랑(1)/초록(2)/빨강(3) 큐브를 voting해 판별한다."""
    h, w = bgr_img.shape[:2]
    margin_y = int(h * (1.0 - roi_ratio) / 2.0)
    margin_x = int(w * (1.0 - roi_ratio) / 2.0)
    roi = bgr_img[margin_y : h - margin_y, margin_x : w - margin_x]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    counts = {}
    for color_id, ranges in COLOR_HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        counts[color_id] = int(np.count_nonzero(mask))

    min_pixels = max(40, int(roi.shape[0] * roi.shape[1] * min_ratio))
    best_color, best_count = max(counts.items(), key=lambda item: item[1])
    if best_count < min_pixels:
        return None

    return best_color


class ColorDetectorNode(Node):
    """/rgb 구독 -> 중앙 큐브 색상 판별 -> /color_id 1회 발행 노드."""

    def __init__(self) -> None:
        super().__init__("color_detector_node")
        self.declare_parameter("rgb_topic", "/rgb")
        self.declare_parameter("color_topic", "/color_id")
        self.declare_parameter("roi_ratio", 0.4)
        self.declare_parameter("min_pixel_ratio", 0.05)
        self.declare_parameter("confirm_frames", 3)

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        self._roi_ratio = self.get_parameter("roi_ratio").get_parameter_value().double_value
        self._min_pixel_ratio = self.get_parameter("min_pixel_ratio").get_parameter_value().double_value
        self._confirm_frames = self.get_parameter("confirm_frames").get_parameter_value().integer_value

        pub_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(Int32, color_topic, pub_qos)
        self._sub = self.create_subscription(Image, rgb_topic, self._on_image, qos_profile_sensor_data)

        self._history: deque[int] = deque(maxlen=max(1, self._confirm_frames))

        self.get_logger().info(f"구독: {rgb_topic} -> 발행: {color_topic} (파랑=1, 초록=2, 빨강=3)")

    def _on_image(self, msg: Image) -> None:
        bgr = image_msg_to_bgr(msg)
        if bgr is None:
            self.get_logger().warn(f"지원하지 않는 인코딩: {msg.encoding}", throttle_duration_sec=5.0)
            return

        detected = detect_center_cube_color(
            bgr,
            roi_ratio=self._roi_ratio,
            min_ratio=self._min_pixel_ratio,
        )

        if detected is None:
            self._history.clear()
            return

        self._history.append(detected)
        if len(self._history) == self._history.maxlen and len(set(self._history)) == 1:
            out = Int32()
            out.data = detected
            self._pub.publish(out)
            self._history.clear()
            label = COLOR_LABELS.get(detected, str(detected))
            self.get_logger().info(f"/color_id 발행 후 초기화: {detected} [{label}]")


def main() -> None:
    rclpy.init()
    node = ColorDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
