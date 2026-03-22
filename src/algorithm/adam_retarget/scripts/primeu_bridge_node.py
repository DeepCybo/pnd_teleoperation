#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from typing import Dict, List, Optional, Sequence
import math

class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        if cutoff <= 0.0 or dt <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, dt: float) -> float:
        if self._x_prev is None or dt <= 0.0:
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        dx = (x - self._x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = self._dx_prev + alpha_d * (dx - self._dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = self._x_prev + alpha * (x - self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

class PrimeUBridgeNode(Node):
    """
    Bridge between Mocap Retargeting output and ros2_control controllers.
    Also handles joint ordering for individual arm controllers.
    
    Supports interpolation to smooth 100Hz input commands to 1000Hz output,
    which is critical for CSP (Cyclic Sync Position) mode EtherCAT drives.
    """
    def __init__(self):
        super().__init__("primeu_bridge_node")

        # Publishing behavior
        # - publish_rate <= 0: publish only on input JointState callback (no interpolation)
        # - publish_rate  > 0: publish at fixed rate with optional interpolation
        self.declare_parameter("publish_rate", 0.0)
        self.declare_parameter("stale_timeout", 0.25)  # seconds
        self.declare_parameter("input_topic", "/primeu/joint_states")

        # Interpolation parameters
        # - interpolation_alpha: 0.0 = no interpolation (step), 1.0 = instant jump
        #   typical: 0.1 means move 10% towards target each publish cycle
        #   For 1000Hz publish, alpha=0.1 gives ~10ms time constant (smooth)
        # - max_velocity_rad_s: limit velocity to prevent jerks (rad/s per joint)
        self.declare_parameter("interpolation_alpha", 0.0)  # 0 = disabled
        self.declare_parameter("max_velocity_rad_s", 1.5)   # rad/s (~86 deg/s)
        self.declare_parameter("max_accel_rad_s2", 6.0)     # rad/s^2
        self.declare_parameter("max_jerk_rad_s3", 60.0)     # rad/s^3
        # OneEuroFilter parameters (recommended defaults)
        self.declare_parameter("one_euro_min_cutoff", 1.0)
        self.declare_parameter("one_euro_beta", 0.007)
        self.declare_parameter("one_euro_d_cutoff", 1.0)

        self.publish_rate: float = float(
            self.get_parameter("publish_rate").get_parameter_value().double_value
        )
        self.input_topic: str = (
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        self.stale_timeout: float = float(
            self.get_parameter("stale_timeout").get_parameter_value().double_value
        )
        self.interpolation_alpha: float = float(
            self.get_parameter("interpolation_alpha").get_parameter_value().double_value
        )
        self.max_velocity_rad_s: float = float(
            self.get_parameter("max_velocity_rad_s").get_parameter_value().double_value
        )
        self.max_accel_rad_s2: float = float(
            self.get_parameter("max_accel_rad_s2").get_parameter_value().double_value
        )
        self.max_jerk_rad_s3: float = float(
            self.get_parameter("max_jerk_rad_s3").get_parameter_value().double_value
        )
        self.one_euro_min_cutoff: float = float(
            self.get_parameter("one_euro_min_cutoff").get_parameter_value().double_value
        )
        self.one_euro_beta: float = float(
            self.get_parameter("one_euro_beta").get_parameter_value().double_value
        )
        self.one_euro_d_cutoff: float = float(
            self.get_parameter("one_euro_d_cutoff").get_parameter_value().double_value
        )
        # Parameters for arm joints (must match hybrid_controllers.yaml order)
        self.declare_parameter(
            "left_arm_joints",
            [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_pitch_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint"
            ],
        )
        self.declare_parameter(
            "right_arm_joints",
            [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_pitch_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint"
            ],
        )
        self.declare_parameter(
            "waist_joints",
            ["waist_yaw_joint"],
        )

        # rclpy Parameter.value is typed as Any; read as string array explicitly.
        self.left_joints: List[str] = list(
            self.get_parameter("left_arm_joints").get_parameter_value().string_array_value
        )
        self.right_joints: List[str] = list(
            self.get_parameter("right_arm_joints").get_parameter_value().string_array_value
        )
        self.waist_joints: List[str] = list(
            self.get_parameter("waist_joints").get_parameter_value().string_array_value
        )

        if not self.left_joints and not self.right_joints and not self.waist_joints:
            self.get_logger().warn("No joints configured; bridge will be idle")

        # Publishers for controllers
        self.left_pub = self.create_publisher(Float64MultiArray, "/left_arm_servo_controller/commands", 10)
        self.right_pub = self.create_publisher(Float64MultiArray, "/right_arm_servo_controller/commands", 10)
        self.waist_pub = self.create_publisher(Float64MultiArray, "/waist_servo_controller/commands", 10)

        # Subscriber to remap output
        self.sub = self.create_subscription(JointState, self.input_topic, self.joint_callback, 10)

        # Target positions (updated by joint_callback)
        self._target_joint_dict: Dict[str, float] = {}
        # Filtered target positions (OneEuroFilter output)
        self._filtered_joint_dict: Dict[str, float] = {}
        # OneEuroFilter per joint
        self._filters: Dict[str, OneEuroFilter] = {}
        self._last_msg_time = None  # rclpy.time.Time
        self._last_missing_warn_time = None  # rclpy.time.Time
        self._initialized = False  # First command received?
        self._last_update_time = None  # rclpy.time.Time

        self._timer = None
        if self.publish_rate and self.publish_rate > 0.0:
            period = 1.0 / self.publish_rate
            self._timer = self.create_timer(period, self._on_timer)

        interp_status = "disabled" if self.interpolation_alpha <= 0.0 else f"alpha={self.interpolation_alpha}"
        self.get_logger().info(
            f"PrimeU Bridge Node Started. Subscribing to {self.input_topic}. "
            f"publish_rate={self.publish_rate}Hz stale_timeout={self.stale_timeout}s "
            f"interpolation={interp_status} "
            f"one_euro(min_cutoff={self.one_euro_min_cutoff}, beta={self.one_euro_beta}, d_cutoff={self.one_euro_d_cutoff}) "
            f"limits(v={self.max_velocity_rad_s}, a={self.max_accel_rad_s2}, j={self.max_jerk_rad_s3})"
        )

    def _get_filter(self, joint_name: str) -> OneEuroFilter:
        filt = self._filters.get(joint_name)
        if filt is None:
            filt = OneEuroFilter(
                self.one_euro_min_cutoff, self.one_euro_beta, self.one_euro_d_cutoff
            )
            self._filters[joint_name] = filt
        return filt

    def _filter_targets(self, dt: float) -> Dict[str, float]:
        if not self._target_joint_dict:
            return {}
        filtered: Dict[str, float] = {}
        use_one_euro = (
            self.one_euro_min_cutoff > 0.0
            or self.one_euro_beta > 0.0
            or self.one_euro_d_cutoff > 0.0
        )
        for joint_name, target_pos in self._target_joint_dict.items():
            if use_one_euro:
                filtered[joint_name] = self._get_filter(joint_name).filter(target_pos, dt)
            else:
                filtered[joint_name] = target_pos
        return filtered

    def _filtered_targets_ready(self) -> bool:
        return bool(self._filtered_joint_dict)

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

        if self.waist_joints:
            waist_cmd = self._pack_arm(joint_dict, self.waist_joints)
            if waist_cmd is not None:
                self.waist_pub.publish(waist_cmd)

    def _on_timer(self) -> None:
        if self._last_msg_time is None:
            return

        age = (self.get_clock().now() - self._last_msg_time).nanoseconds / 1e9
        if age > self.stale_timeout:
            # Stop commanding if upstream data is stale.
            return

        now = self.get_clock().now()
        if self._last_update_time is None:
            dt = 0.0
        else:
            dt = (now - self._last_update_time).nanoseconds / 1e9
        self._last_update_time = now

        self._filtered_joint_dict = self._filter_targets(dt)
        if self._filtered_targets_ready():
            self._publish_from_joint_dict(self._filtered_joint_dict)

    def joint_callback(self, msg: JointState):
        # Create a mapping for quick lookup
        joint_dict = {name: pos for name, pos in zip(msg.name, msg.position)}

        self._target_joint_dict = joint_dict
        self._last_msg_time = self.get_clock().now()

        # If not using timer-based publish, publish immediately on input (no interpolation).
        if not self.publish_rate or self.publish_rate <= 0.0:
            now = self._last_msg_time
            if self._last_update_time is None:
                dt = 0.0
            else:
                dt = (now - self._last_update_time).nanoseconds / 1e9
            self._last_update_time = now

            self._filtered_joint_dict = self._filter_targets(dt)
            if self._filtered_targets_ready():
                self._publish_from_joint_dict(self._filtered_joint_dict)

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
