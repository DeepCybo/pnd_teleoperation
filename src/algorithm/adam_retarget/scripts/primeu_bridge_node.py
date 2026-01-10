#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from typing import Dict, List, Optional, Sequence

class PrimeUBridgeNode(Node):
    """
    Bridge between Mocap Retargeting output and ros2_control controllers.
    Also handles joint ordering for individual arm controllers.
    """
    def __init__(self):
        super().__init__("primeu_bridge_node")

        # Publishing behavior
        # - publish_rate <= 0: publish only on input JointState callback
        # - publish_rate  > 0: publish at fixed rate using last received JointState
        self.declare_parameter("publish_rate", 0.0)
        self.declare_parameter("stale_timeout", 0.25)  # seconds
        self.publish_rate: float = float(
            self.get_parameter("publish_rate").get_parameter_value().double_value
        )
        self.stale_timeout: float = float(
            self.get_parameter("stale_timeout").get_parameter_value().double_value
        )

        # Parameters for arm joints (must match hybrid_controllers.yaml order)
        self.declare_parameter("left_arm_joints", [
            "arm_left_shoulder_pitch_joint", "arm_left_shoulder_roll_joint",
            "arm_left_shoulder_yaw_joint", "arm_left_elbow_pitch_joint",
            "arm_left_wrist_yaw_joint", "arm_left_wrist_roll_joint", "arm_left_wrist_pitch_joint"
        ])
        self.declare_parameter("right_arm_joints", [
            "arm_right_shoulder_pitch_joint", "arm_right_shoulder_roll_joint",
            "arm_right_shoulder_yaw_joint", "arm_right_elbow_pitch_joint",
            "arm_right_wrist_yaw_joint", "arm_right_wrist_roll_joint", "arm_right_wrist_pitch_joint"
        ])

        # rclpy Parameter.value is typed as Any; read as string array explicitly.
        self.left_joints: List[str] = list(
            self.get_parameter("left_arm_joints").get_parameter_value().string_array_value
        )
        self.right_joints: List[str] = list(
            self.get_parameter("right_arm_joints").get_parameter_value().string_array_value
        )

        if not self.left_joints and not self.right_joints:
            self.get_logger().warn("No arm joints configured; bridge will be idle")

        # Publishers for controllers
        self.left_pub = self.create_publisher(Float64MultiArray, "/left_arm_servo_controller/commands", 10)
        self.right_pub = self.create_publisher(Float64MultiArray, "/right_arm_servo_controller/commands", 10)

        # Subscriber to remap output
        self.sub = self.create_subscription(JointState, "/primeu/joint_states", self.joint_callback, 10)

        self._last_joint_dict: Dict[str, float] = {}
        self._last_msg_time = None  # rclpy.time.Time
        self._last_missing_warn_time = None  # rclpy.time.Time

        self._timer = None
        if self.publish_rate and self.publish_rate > 0.0:
            period = 1.0 / self.publish_rate
            self._timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"PrimeU Bridge Node Started. Subscribing to /primeu/joint_states. "
            f"publish_rate={self.publish_rate}Hz stale_timeout={self.stale_timeout}s"
        )

    def _pack_arm(
        self, joint_dict: Dict[str, float], joint_list: Sequence[str]
    ) -> Optional[Float64MultiArray]:
        # Require all joints to be present to avoid accidentally commanding 0.0
        missing = [j for j in joint_list if j not in joint_dict]
        if missing:
            now = self.get_clock().now()
            if (
                self._last_missing_warn_time is None
                or (now - self._last_missing_warn_time).nanoseconds / 1e9 > 2.0
            ):
                self._last_missing_warn_time = now
                self.get_logger().warn(
                    f"Missing {len(missing)} joints from input JointState; skipping publish. Example: {missing[:3]}"
                )
            return None

        cmd = Float64MultiArray()
        cmd.data = [joint_dict[j] for j in joint_list]
        return cmd

    def _publish_from_joint_dict(self, joint_dict: Dict[str, float]) -> None:
        if not joint_dict:
            return

        if self.left_joints:
            left_cmd = self._pack_arm(joint_dict, self.left_joints)
            if left_cmd is not None:
                self.left_pub.publish(left_cmd)

        if self.right_joints:
            right_cmd = self._pack_arm(joint_dict, self.right_joints)
            if right_cmd is not None:
                self.right_pub.publish(right_cmd)

    def _on_timer(self) -> None:
        if self._last_msg_time is None:
            return

        age = (self.get_clock().now() - self._last_msg_time).nanoseconds / 1e9
        if age > self.stale_timeout:
            # Stop commanding if upstream data is stale.
            return

        self._publish_from_joint_dict(self._last_joint_dict)

    def joint_callback(self, msg: JointState):
        # Create a mapping for quick lookup
        joint_dict = {name: pos for name, pos in zip(msg.name, msg.position)}

        self._last_joint_dict = joint_dict
        self._last_msg_time = self.get_clock().now()

        # If not using timer-based publish, publish immediately on input.
        if not self.publish_rate or self.publish_rate <= 0.0:
            self._publish_from_joint_dict(joint_dict)

def main():
    rclpy.init()
    node = PrimeUBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
