#!/usr/bin/env python3

import os
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import pinocchio as pin
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from tf2_ros import Buffer, TransformException, TransformListener


class HeadPinocchioIk(Node):
    def __init__(self) -> None:
        super().__init__("head_pinocchio_ik")

        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("command_topic", "/neck_servo_controller/commands")
        self.declare_parameter("visualization_source_topic", "/primeu/remap_joint_states")
        self.declare_parameter("visualization_joint_topic", "/primeu/joint_states")
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("tf_timeout_sec", 0.05)
        self.declare_parameter("mocap_neck_frame", "noitom/Neck")
        self.declare_parameter("mocap_head_frame", "noitom/Head")
        self.declare_parameter("robot_base_frame", "chest_link")
        self.declare_parameter("robot_tip_frame", "neck_pitch_link")
        self.declare_parameter(
            "neck_joint_names",
            ["neck_yaw_joint", "neck_roll_joint", "neck_pitch_joint"],
        )
        self.declare_parameter("urdf_package", "primeu_description")
        self.declare_parameter(
            "urdf_relative_path",
            "urdf/primeu_robot_with_wuji_hands.urdf",
        )
        self.declare_parameter("solver_max_nfev", 25)
        self.declare_parameter("solver_ftol", 1e-6)
        self.declare_parameter("solver_xtol", 1e-6)
        self.declare_parameter("command_smoothing_alpha", 1.0)
        self.declare_parameter("auto_calibrate", True)
        self.declare_parameter("calibration_roll", 0.0)
        self.declare_parameter("calibration_pitch", 0.0)
        self.declare_parameter("calibration_yaw", 0.0)
        self.declare_parameter(
            "mocap_axes_correction_rpy",
            [-1.57079633, -1.57079633, 0.0],
        )
        self.declare_parameter(
            "mocap_axes_sign",
            [1.0, 1.0, -1.0],
        )
        self.declare_parameter("publish_debug_topics", True)
        self.declare_parameter("residual_warn_threshold_rad", 0.15)

        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.command_topic = self.get_parameter("command_topic").value
        self.visualization_source_topic = self.get_parameter(
            "visualization_source_topic"
        ).value
        self.visualization_joint_topic = self.get_parameter(
            "visualization_joint_topic"
        ).value
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.tf_timeout = Duration(seconds=float(self.get_parameter("tf_timeout_sec").value))
        self.mocap_neck_frame = self.get_parameter("mocap_neck_frame").value
        self.mocap_head_frame = self.get_parameter("mocap_head_frame").value
        self.robot_base_frame = self.get_parameter("robot_base_frame").value
        self.robot_tip_frame = self.get_parameter("robot_tip_frame").value
        self.neck_joint_names = list(self.get_parameter("neck_joint_names").value)
        self.solver_max_nfev = int(self.get_parameter("solver_max_nfev").value)
        self.solver_ftol = float(self.get_parameter("solver_ftol").value)
        self.solver_xtol = float(self.get_parameter("solver_xtol").value)
        self.command_smoothing_alpha = float(
            self.get_parameter("command_smoothing_alpha").value
        )
        self.auto_calibrate = bool(self.get_parameter("auto_calibrate").value)
        self.publish_debug_topics = bool(
            self.get_parameter("publish_debug_topics").value
        )
        self.residual_warn_threshold_rad = float(
            self.get_parameter("residual_warn_threshold_rad").value
        )

        self.command_smoothing_alpha = float(
            np.clip(self.command_smoothing_alpha, 0.0, 1.0)
        )

        calibration_rotation = Rotation.from_euler(
            "xyz",
            [
                float(self.get_parameter("calibration_roll").value),
                float(self.get_parameter("calibration_pitch").value),
                float(self.get_parameter("calibration_yaw").value),
            ],
        )
        self._manual_calibration_rotation = calibration_rotation.as_matrix()
        self._calibration_rotation = self._manual_calibration_rotation.copy()
        self._calibrated = not self.auto_calibrate

        axes_rpy = list(self.get_parameter("mocap_axes_correction_rpy").value)
        self._mocap_axes_fix = Rotation.from_euler("xyz", axes_rpy).as_matrix()

        self._mocap_axes_sign = np.array(
            self.get_parameter("mocap_axes_sign").value, dtype=float
        )
        self.get_logger().info(
            f"Mocap axes sign: X(yaw)={self._mocap_axes_sign[0]:+.0f}, "
            f"Y(roll)={self._mocap_axes_sign[1]:+.0f}, "
            f"Z(pitch)={self._mocap_axes_sign[2]:+.0f}"
        )

        urdf_package = self.get_parameter("urdf_package").value
        urdf_relative_path = self.get_parameter("urdf_relative_path").value
        urdf_path = os.path.join(
            get_package_share_directory(urdf_package), urdf_relative_path
        )

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.neutral_q = pin.neutral(self.model)
        self.base_frame_id = self.model.getFrameId(self.robot_base_frame)
        self.tip_frame_id = self.model.getFrameId(self.robot_tip_frame)
        if self.base_frame_id >= len(self.model.frames):
            raise RuntimeError(f"Frame '{self.robot_base_frame}' not found in URDF")
        if self.tip_frame_id >= len(self.model.frames):
            raise RuntimeError(f"Frame '{self.robot_tip_frame}' not found in URDF")

        self.scalar_joint_q_index: Dict[str, int] = {}
        for joint_id in range(1, self.model.njoints):
            joint_name = self.model.names[joint_id]
            joint_model = self.model.joints[joint_id]
            if joint_model.nq == 1:
                self.scalar_joint_q_index[joint_name] = joint_model.idx_q

        missing_joint_names = [
            name for name in self.neck_joint_names if name not in self.scalar_joint_q_index
        ]
        if missing_joint_names:
            raise RuntimeError(
                f"Could not locate neck joints in URDF model: {missing_joint_names}"
            )

        self.neck_q_indices = np.array(
            [self.scalar_joint_q_index[name] for name in self.neck_joint_names],
            dtype=int,
        )
        self.lower_limits = np.asarray(
            self.model.lowerPositionLimit[self.neck_q_indices], dtype=float
        )
        self.upper_limits = np.asarray(
            self.model.upperPositionLimit[self.neck_q_indices], dtype=float
        )

        self.joint_positions: Dict[str, float] = {}
        self.latest_visualization_source: Optional[JointState] = None
        self.last_command: Optional[np.ndarray] = None
        self._log_times: Dict[str, float] = {}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        joint_state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.command_pub = self.create_publisher(Float64MultiArray, self.command_topic, 10)
        self.visualization_joint_pub = self.create_publisher(
            JointState, self.visualization_joint_topic, 10
        )
        self.debug_joint_pub = self.create_publisher(
            JointState, "/primeu/head_ik/joint_states", 10
        )
        self.debug_target_pub = self.create_publisher(
            PoseStamped, "/primeu/head_ik/target_pose", 10
        )
        self.active_pub = self.create_publisher(Bool, "/primeu/head_ik/active", 10)

        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_callback,
            joint_state_qos,
        )
        self.create_subscription(
            JointState,
            self.visualization_source_topic,
            self._visualization_source_callback,
            10,
        )

        period = 1.0 / self.publish_rate if self.publish_rate > 0.0 else 0.01
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            "Head Pinocchio IK ready. "
            f"TF {self.mocap_neck_frame}->{self.mocap_head_frame}, "
            f"robot {self.robot_base_frame}->{self.robot_tip_frame}, "
            f"command_topic={self.command_topic}"
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        self.joint_positions.update(
            {name: pos for name, pos in zip(msg.name, msg.position)}
        )

    def _visualization_source_callback(self, msg: JointState) -> None:
        self.latest_visualization_source = msg

    def _throttled_log(
        self, key: str, message: str, level: str = "warn", period_sec: float = 1.0
    ) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec - self._log_times.get(key, -1e9) < period_sec:
            return
        self._log_times[key] = now_sec
        getattr(self.get_logger(), level)(message)

    def _build_full_q(self, neck_override: Optional[np.ndarray] = None) -> np.ndarray:
        q = self.neutral_q.copy()
        for joint_name, idx_q in self.scalar_joint_q_index.items():
            pos = self.joint_positions.get(joint_name)
            if pos is not None:
                q[idx_q] = pos
        if neck_override is not None:
            q[self.neck_q_indices] = neck_override
        return q

    def _current_neck_q(self) -> Optional[np.ndarray]:
        if all(name in self.joint_positions for name in self.neck_joint_names):
            return np.array(
                [self.joint_positions[name] for name in self.neck_joint_names],
                dtype=float,
            )
        if self.last_command is not None:
            return self.last_command.copy()
        return None

    def _relative_rotation(self, neck_q: np.ndarray) -> np.ndarray:
        q_full = self._build_full_q(neck_override=neck_q)
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)
        base_rotation = self.data.oMf[self.base_frame_id].rotation
        tip_rotation = self.data.oMf[self.tip_frame_id].rotation
        return base_rotation.T @ tip_rotation

    def _solve_neck_ik(
        self, target_rotation: np.ndarray, q_seed: np.ndarray
    ) -> tuple[np.ndarray, float]:
        q_seed = np.clip(q_seed, self.lower_limits, self.upper_limits)

        def residual(neck_q: np.ndarray) -> np.ndarray:
            current_rotation = self._relative_rotation(neck_q)
            rotation_error = target_rotation @ current_rotation.T
            return np.asarray(pin.log3(rotation_error), dtype=float)

        result = least_squares(
            residual,
            q_seed,
            bounds=(self.lower_limits, self.upper_limits),
            method="trf",
            max_nfev=self.solver_max_nfev,
            ftol=self.solver_ftol,
            xtol=self.solver_xtol,
            gtol=self.solver_ftol,
        )

        solved = np.clip(result.x, self.lower_limits, self.upper_limits)
        final_residual = float(np.linalg.norm(residual(solved)))
        if not result.success:
            self._throttled_log(
                "ik_solver_failure",
                f"Head IK solver did not fully converge: {result.message}",
            )
        return solved, final_residual

    def _lookup_mocap_relative_pose(self) -> tuple[np.ndarray, np.ndarray]:
        transform = self.tf_buffer.lookup_transform(
            self.mocap_neck_frame,
            self.mocap_head_frame,
            rclpy.time.Time(),
            timeout=self.tf_timeout,
        )
        translation = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ],
            dtype=float,
        )
        rotation = Rotation.from_quat(
            [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
        ).as_matrix()
        return translation, rotation

    def _publish_joint_state(
        self, stamp, topic_pub, neck_command: np.ndarray
    ) -> None:
        joint_msg = JointState()
        joint_msg.header.stamp = stamp
        joint_msg.name = list(self.neck_joint_names)
        joint_msg.position = [float(v) for v in neck_command]
        topic_pub.publish(joint_msg)

    def _publish_visualization_joint_state(self, stamp, neck_command: np.ndarray) -> None:
        if self.latest_visualization_source is None:
            self._publish_joint_state(stamp, self.visualization_joint_pub, neck_command)
            return

        merged = deepcopy(self.latest_visualization_source)
        merged.header.stamp = stamp

        name_to_index = {name: idx for idx, name in enumerate(merged.name)}
        for joint_name, joint_value in zip(self.neck_joint_names, neck_command):
            idx = name_to_index.get(joint_name)
            if idx is None:
                merged.name.append(joint_name)
                merged.position.append(float(joint_value))
                if len(merged.velocity) == len(merged.name) - 1:
                    merged.velocity.append(0.0)
                if len(merged.effort) == len(merged.name) - 1:
                    merged.effort.append(0.0)
                continue

            if idx < len(merged.position):
                merged.position[idx] = float(joint_value)
            else:
                merged.position.extend([0.0] * (idx + 1 - len(merged.position)))
                merged.position[idx] = float(joint_value)

            if merged.velocity and idx < len(merged.velocity):
                merged.velocity[idx] = 0.0
            if merged.effort and idx < len(merged.effort):
                merged.effort[idx] = 0.0

        self.visualization_joint_pub.publish(merged)

    def _publish_debug(
        self, translation: np.ndarray, target_rotation: np.ndarray, neck_command: np.ndarray
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        self._publish_visualization_joint_state(stamp, neck_command)

        if self.publish_debug_topics:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = self.robot_base_frame
            pose_msg.pose.position.x = float(translation[0])
            pose_msg.pose.position.y = float(translation[1])
            pose_msg.pose.position.z = float(translation[2])
            quat = Rotation.from_matrix(target_rotation).as_quat()
            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])
            self.debug_target_pub.publish(pose_msg)
            self._publish_joint_state(stamp, self.debug_joint_pub, neck_command)

        active_msg = Bool()
        active_msg.data = True
        self.active_pub.publish(active_msg)

    def _publish_inactive(self) -> None:
        active_msg = Bool()
        active_msg.data = False
        self.active_pub.publish(active_msg)

    def _on_timer(self) -> None:
        q_seed = self._current_neck_q()
        if q_seed is None:
            self._publish_inactive()
            self._throttled_log(
                "missing_joint_state",
                "Waiting for neck joint states before solving head IK.",
            )
            return

        try:
            translation, mocap_rotation_raw = self._lookup_mocap_relative_pose()
        except TransformException as exc:
            self._publish_inactive()
            self._throttled_log(
                "tf_lookup_failed",
                f"Head IK TF lookup failed for {self.mocap_neck_frame}->{self.mocap_head_frame}: {exc}",
            )
            return

        mocap_rotation = self._mocap_axes_fix @ mocap_rotation_raw @ self._mocap_axes_fix.T

        rotvec = Rotation.from_matrix(mocap_rotation).as_rotvec()
        rotvec *= self._mocap_axes_sign
        mocap_rotation = Rotation.from_rotvec(rotvec).as_matrix()

        if not self._calibrated:
            current_robot_rotation = self._relative_rotation(q_seed)
            self._calibration_rotation = current_robot_rotation @ mocap_rotation.T
            self._calibrated = True
            self.get_logger().info("Head IK auto-calibration captured current neutral pose.")

        target_rotation = (
            self._manual_calibration_rotation
            @ self._calibration_rotation
            @ mocap_rotation
        )
        solved_q, residual_norm = self._solve_neck_ik(target_rotation, q_seed)

        if residual_norm > self.residual_warn_threshold_rad:
            self._throttled_log(
                "large_residual",
                f"Head IK residual is high ({residual_norm:.3f} rad). Check calibration or axes alignment.",
            )

        if self.last_command is None or self.command_smoothing_alpha >= 1.0:
            command_q = solved_q
        else:
            command_q = self.last_command + self.command_smoothing_alpha * (
                solved_q - self.last_command
            )

        clipped_q = np.clip(command_q, self.lower_limits, self.upper_limits)
        if np.any(np.abs(clipped_q - command_q) > 1e-6):
            self._throttled_log(
                "joint_limit_clip",
                "Head IK command clipped by neck joint limits.",
            )
        command_q = clipped_q
        self.last_command = command_q

        command_msg = Float64MultiArray()
        command_msg.data = [float(v) for v in command_q]
        self.command_pub.publish(command_msg)

        self._publish_debug(translation, target_rotation, command_q)


def main() -> None:
    rclpy.init()
    node = HeadPinocchioIk()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
