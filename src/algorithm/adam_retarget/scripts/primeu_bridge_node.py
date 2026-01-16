#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from typing import Dict, List, Optional, Sequence
import math

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

        # Interpolation parameters
        # - interpolation_alpha: 0.0 = no interpolation (step), 1.0 = instant jump
        #   typical: 0.1 means move 10% towards target each publish cycle
        #   For 1000Hz publish, alpha=0.1 gives ~10ms time constant (smooth)
        # - max_velocity_rad_s: limit velocity to prevent jerks (rad/s per joint)
        self.declare_parameter("interpolation_alpha", 0.0)  # 0 = disabled
        self.declare_parameter("max_velocity_rad_s", 3.0)   # rad/s (~170 deg/s)

        self.publish_rate: float = float(
            self.get_parameter("publish_rate").get_parameter_value().double_value
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

        # Parameters for arm joints (must match hybrid_controllers.yaml order)
        self.declare_parameter("left_arm_joints", [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
            "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint"
        ])
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

        # Target positions (updated by joint_callback)
        self._target_joint_dict: Dict[str, float] = {}
        # Current interpolated positions (updated by timer, sent to controller)
        self._current_joint_dict: Dict[str, float] = {}
        self._last_msg_time = None  # rclpy.time.Time
        self._last_missing_warn_time = None  # rclpy.time.Time
        self._initialized = False  # First command received?

        # Compute max step per publish cycle
        self._max_step_per_cycle = 0.0
        if self.publish_rate > 0.0:
            self._max_step_per_cycle = self.max_velocity_rad_s / self.publish_rate

        self._timer = None
        if self.publish_rate and self.publish_rate > 0.0:
            period = 1.0 / self.publish_rate
            self._timer = self.create_timer(period, self._on_timer)

        interp_status = "disabled" if self.interpolation_alpha <= 0.0 else f"alpha={self.interpolation_alpha}"
        self.get_logger().info(
            f"PrimeU Bridge Node Started. Subscribing to /primeu/joint_states. "
            f"publish_rate={self.publish_rate}Hz stale_timeout={self.stale_timeout}s "
            f"interpolation={interp_status} max_vel={self.max_velocity_rad_s} rad/s"
        )

    def _interpolate_step(self) -> None:
        """
        Move _current_joint_dict towards _target_joint_dict by one interpolation step.
        Uses exponential smoothing with velocity limiting.
        """
        if not self._target_joint_dict:
            return

        if not self._initialized:
            # First time: snap to target
            self._current_joint_dict = dict(self._target_joint_dict)
            self._initialized = True
            return

        alpha = self.interpolation_alpha
        max_step = self._max_step_per_cycle

        for joint_name, target_pos in self._target_joint_dict.items():
            current_pos = self._current_joint_dict.get(joint_name, target_pos)

            if alpha <= 0.0:
                # No interpolation, just use target directly
                new_pos = target_pos
            else:
                # Exponential smoothing: new = current + alpha * (target - current)
                delta = target_pos - current_pos

                # Apply alpha
                step = alpha * delta

                # Clamp by max velocity
                if max_step > 0.0 and abs(step) > max_step:
                    step = math.copysign(max_step, step)

                new_pos = current_pos + step

            self._current_joint_dict[joint_name] = new_pos

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

        # Perform interpolation step if enabled
        if self.interpolation_alpha > 0.0:
            self._interpolate_step()
            self._publish_from_joint_dict(self._current_joint_dict)
        else:
            self._publish_from_joint_dict(self._target_joint_dict)

    def joint_callback(self, msg: JointState):
        # Create a mapping for quick lookup
        joint_dict = {name: pos for name, pos in zip(msg.name, msg.position)}

        self._target_joint_dict = joint_dict
        self._last_msg_time = self.get_clock().now()

        # If not using timer-based publish, publish immediately on input (no interpolation).
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
