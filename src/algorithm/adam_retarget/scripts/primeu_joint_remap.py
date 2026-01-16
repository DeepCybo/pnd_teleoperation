#!/usr/bin/env python3

import json
from dataclasses import dataclass
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


@dataclass(frozen=True)
class JointMap:
    output: str
    scale: float = 1.0
    offset: float = 0.0


class PrimeUJointRemap(Node):
    def __init__(self):
        super().__init__("primeu_joint_remap")

        self.declare_parameter("input_topic", "/adam/joint_states")
        self.declare_parameter("output_topic", "/primeu/joint_states")
        self.declare_parameter("mapping_file", "")

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        mapping_file = self.get_parameter("mapping_file").get_parameter_value().string_value

        self._default_map: Dict[str, JointMap] = {
            # Waist
            "dof_pos/waistYaw": JointMap(output="waist_yaw_joint"),
            "dof_pos/waistRoll": JointMap(output="waist_roll_joint"),
            "dof_pos/waistPitch": JointMap(output="waist_pitch_joint"),

            # Left arm
            "dof_pos/shoulderPitch_Left": JointMap(output="left_shoulder_pitch_joint"),
            "dof_pos/shoulderRoll_Left": JointMap(output="left_shoulder_roll_joint"),
            "dof_pos/shoulderYaw_Left": JointMap(output="left_shoulder_yaw_joint"),
            "dof_pos/elbow_Left": JointMap(output="left_elbow_pitch_joint"),
            "dof_pos/wristYaw_Left": JointMap(output="left_wrist_yaw_joint"),
            "dof_pos/wristPitch_Left": JointMap(output="left_wrist_pitch_joint"),
            "dof_pos/wristRoll_Left": JointMap(output="left_wrist_roll_joint"),

            # Right arm
            "dof_pos/shoulderPitch_Right": JointMap(output="right_shoulder_pitch_joint"),
            "dof_pos/shoulderRoll_Right": JointMap(output="right_shoulder_roll_joint"),
            "dof_pos/shoulderYaw_Right": JointMap(output="right_shoulder_yaw_joint"),
            "dof_pos/elbow_Right": JointMap(output="right_elbow_pitch_joint"),
            "dof_pos/wristYaw_Right": JointMap(output="right_wrist_yaw_joint"),
            "dof_pos/wristPitch_Right": JointMap(output="right_wrist_pitch_joint"),
            "dof_pos/wristRoll_Right": JointMap(output="right_wrist_roll_joint"),
        }

        self._map: Dict[str, JointMap] = self._load_mapping_file(mapping_file) or self._default_map

        self._pub = self.create_publisher(JointState, output_topic, 10)
        self._sub = self.create_subscription(JointState, input_topic, self._on_joint_state, 10)

        if mapping_file:
            self.get_logger().info(f"Loaded mapping_file: {mapping_file}")
        self.get_logger().info(f"Remapping JointState: {input_topic} -> {output_topic}")

    def _load_mapping_file(self, path: str) -> Optional[Dict[str, JointMap]]:
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            joints = data.get("joints", {})
            result: Dict[str, JointMap] = {}
            for input_name, spec in joints.items():
                if not isinstance(spec, dict):
                    continue
                output = spec.get("output")
                if not output:
                    continue
                scale = float(spec.get("scale", 1.0))
                offset = float(spec.get("offset", 0.0))
                result[str(input_name)] = JointMap(output=str(output), scale=scale, offset=offset)
            if not result:
                self.get_logger().warn("mapping_file loaded but no valid joints found; using defaults")
                return None
            return result
        except Exception as e:
            self.get_logger().warn(f"Failed to load mapping_file '{path}': {e}; using defaults")
            return None

    def _on_joint_state(self, msg: JointState) -> None:
        out = JointState()
        out.header = msg.header

        has_velocity = len(msg.velocity) == len(msg.name)
        has_effort = len(msg.effort) == len(msg.name)

        out_names = []
        out_positions = []
        out_velocities = [] if has_velocity else None
        out_efforts = [] if has_effort else None

        for idx, in_name in enumerate(msg.name):
            m = self._map.get(in_name)
            if m is None:
                continue

            out_names.append(m.output)
            in_pos = msg.position[idx] if idx < len(msg.position) else 0.0
            out_positions.append(m.scale * in_pos + m.offset)

            if has_velocity:
                out_velocities.append(msg.velocity[idx])
            if has_effort:
                out_efforts.append(msg.effort[idx])

        out.name = out_names
        out.position = out_positions
        if has_velocity:
            out.velocity = out_velocities
        if has_effort:
            out.effort = out_efforts

        self._pub.publish(out)


def main():
    rclpy.init()
    node = PrimeUJointRemap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
