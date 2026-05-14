import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ld = LaunchDescription()
    enable_mocap_visual = LaunchConfiguration("enable_mocap_visual")

    visual_declare = DeclareLaunchArgument(
        "visual",
        default_value="foxglove",
        description="Choose visualization tool: rviz2 or foxglove",
        choices=["rviz2", "foxglove"],
    )
    ld.add_entity(visual_declare)
    enable_head_ik_declare = DeclareLaunchArgument(
        "enable_head_ik",
        default_value="false",
        description="Enable Noitom head tracking -> PrimeU raw human neck IK node",
    )
    ld.add_entity(enable_head_ik_declare)
    ld.add_entity(
        DeclareLaunchArgument(
            "enable_mocap_visual",
            default_value="false",
            description="Enable the mocap shadow robot visualization model",
        )
    )
    ld.add_entity(
        DeclareLaunchArgument(
            "retarget_rate",
            default_value="50.0",
            description="ADAM retarget solver rate in Hz",
        )
    )
    ld.add_entity(
        DeclareLaunchArgument(
            "body_command_rate",
            default_value="100.0",
            description="Arm raw command publish rate in Hz",
        )
    )
    ld.add_entity(
        DeclareLaunchArgument(
            "waist_retarget_rate",
            default_value="50.0",
            description="Waist RPY bridge publish rate in Hz",
        )
    )
    ld.add_entity(
        DeclareLaunchArgument(
            "head_ik_rate",
            default_value="30.0",
            description="Head IK raw command publish rate in Hz",
        )
    )
    ld.add_entity(
        DeclareLaunchArgument(
            "mocap_visual_rate",
            default_value="30.0",
            description="Mocap robot visualization JointState publish rate in Hz",
        )
    )

    # URDF (PrimeU)
    package_name = "primeu_description"
    urdf_name = "urdf/primeu_robot_with_wuji_hands.urdf"
    urdf_pkg_share = FindPackageShare(package=package_name).find(package_name)
    urdf_model_path = os.path.join(urdf_pkg_share, urdf_name)

    with open(urdf_model_path, "r") as infp:
        robot_desc = infp.read()

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher_mocap",
        output="screen",
        parameters=[{
            "robot_description": robot_desc,
            "frame_prefix": "mocap/"
        }],
        remappings=[
            ("/joint_states", "/primeu/mocap_visual_joint_states"),
            ("/robot_description", "/mocap/robot_description"),
        ],
        condition=IfCondition(enable_mocap_visual),
    )

    # 3. Controller Bridge (Connects mocap JointState to raw human commands)
    #
    # NOTE: this launch publishes body retarget outputs only.  The PrimeU
    # bringup's human trajectory bridge snapshots these raw groups into
    # /primeu/control/human_trajectory for the unified controller.
    #
    # The waist output from this bridge is remapped to a dead topic so the
    # waist path is only driven by the parallel IK pipeline:
    #     /primeu/remap_joint_states (roll/pitch/yaw)
    #         -> waist_retarget_bridge
    #         -> /waist_parallel_ik_node/target_rpy
    #         -> waist_parallel_ik_node
    #         -> /primeu/control/human/raw/waist_commands
    # Letting this bridge also publish a waist motor group would race against
    # the IK node on the same raw human group.
    # (We cannot pass `waist_joints: []` here: ROS 2 Jazzy's launch rejects
    #  empty-list parameter values as an untyped tuple.)
    primeu_controller_bridge_node = Node(
        package="adam_retarget",
        executable="primeu_bridge_node.py",
        name="primeu_controller_bridge",
        output="screen",
        parameters=[
            {
                # Mocap/retarget data arrives at 100 Hz. The 1 kHz hardware
                # loop holds the latest command between ROS topic updates, so
                # this bridge should not publish at motor-loop rate.
                "input_topic": "/primeu/remap_joint_states",
                "publish_rate": ParameterValue(
                    LaunchConfiguration("body_command_rate"), value_type=float
                ),
                "stale_timeout": 0.05,
                # OneEuroFilter
                "one_euro_min_cutoff": 1.0,
                "one_euro_beta": 0.007,
                "one_euro_d_cutoff": 1.0,
                # Motion limits (conservative)
                "max_velocity_rad_s": 1.5,
                "max_accel_rad_s2": 6.0,
                "max_jerk_rad_s3": 60.0,
                "enable_motion_limits": False,
                # Legacy interpolation parameter (kept for compatibility)
                "interpolation_alpha": 0.0,
            }
        ],
        remappings=[
            (
                "/left_arm_servo_controller/commands",
                "/primeu/control/human/raw/left_arm_commands",
            ),
            (
                "/right_arm_servo_controller/commands",
                "/primeu/control/human/raw/right_arm_commands",
            ),
            (
                "/waist_servo_controller/commands",
                "/primeu/control/human/raw/waist_commands_disabled_by_parallel_ik",
            ),
        ],
    )

    # Waist retarget bridge: taps the retargeted joint state, extracts
    # waist_roll_passive_joint / waist_pitch_passive_joint / waist_yaw_joint,
    # clamps / sign-flips them, and publishes a Vector3(rad) to the waist
    # parallel-link IK node.
    #
    # Smoothing (OneEuro) is intentionally disabled here: the IK node owns
    # waist smoothing at the 100 Hz mocap/retarget rate.  The 1 kHz control
    # loop holds the latest command instead of asking IK to solve at motor
    # frequency.
    waist_retarget_bridge_node = Node(
        package="primeu_waist_ik",
        executable="waist_retarget_bridge.py",
        name="waist_retarget_bridge",
        output="screen",
        parameters=[
            {
                "input_topic": "/primeu/remap_joint_states",
                "output_topic": "/waist_parallel_ik_node/target_rpy",
                "roll_joint": "waist_roll_passive_joint",
                "pitch_joint": "waist_pitch_passive_joint",
                "yaw_joint": "waist_yaw_joint",
                "publish_rate": ParameterValue(
                    LaunchConfiguration("waist_retarget_rate"), value_type=float
                ),
                "stale_timeout": 0.1,
                # OneEuro disabled here; IK node does the smoothing.
                "one_euro_min_cutoff": 0.0,
                "one_euro_beta": 0.0,
                "one_euro_d_cutoff": 0.0,
                # Match the waist_parallel_ik_node's passive joint ranges.
                "max_roll_rad": 0.5236,   # ~30 deg
                "max_pitch_rad": 0.5236,  # ~30 deg
                "max_yaw_rad": 3.1416,    # ~180 deg
                # The retarget source's waist roll / pitch are mirrored
                # relative to the MJCF-IK convention used by
                # waist_parallel_ik_node, so flip their signs here.
                "roll_sign": -1.0,
                "pitch_sign": -1.0,
                "yaw_sign": 1.0,
            }
        ],
    )

    # --- TF roots ---
    # Global root is `world`.
    # Noitom raw data is published in `world_noitom_yup`.
    # We provide a z-up root `world_noitom` and connect it to the raw y-up frame.

    world_to_world_noitom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_world_noitom",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "world", "world_noitom"],
    )

    noitom_yup_to_zup = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="noitom_yup_to_zup",
        output="screen",
        # Quaternion (0.7071,0,0,0.7071) == +90deg about X.
        # This provides a fixed relation between z-up and y-up roots.
        arguments=[
            "0",
            "0",
            "0",
            "0.707106781",
            "0",
            "0",
            "0.707106781",
            "world_noitom",
            "world_noitom_yup",
        ],
    )

    # Anchor the mocap robot TF tree (frame_prefix=mocap/) into the global world.
    mocap_robot_anchor = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="mocap_robot_anchor",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "world", "mocap/body_base_link"],
        condition=IfCondition(enable_mocap_visual),
    )

    # Retarget (reuse ADAM-U noitom solver) + publish to /adam/joint_states
    adam_retarget_pkg_share = FindPackageShare(package="adam_retarget").find(
        "adam_retarget"
    )
    default_config_json_path = os.path.join(
        adam_retarget_pkg_share,
        "opti_config/adam_u_codegen_test/Adam_U_Noitom_Deploy_opti.json",
    )

    adam_retarget_node = Node(
        package="adam_retarget",
        executable="adam_retarget",
        name="adam_retarget",
        output="screen",
        parameters=[
            {
                "base_frame": "world_noitom",
                "bone_frame_prefix": "noitom/",
                "control_loop_rate": ParameterValue(
                    LaunchConfiguration("retarget_rate"), value_type=float
                ),
                "config_json_path": default_config_json_path,
                "warm_start_trig_timeout": 0.2,
                "warm_start_duration": 5.0,
                "warm_start_slowdown_ratio": 0.1,
            },
        ],
        remappings=[
            ("/joint_states", "/adam/joint_states"),
        ],
    )

    # Remap ADAM joint labels -> PrimeU joint names (+ scale/offset) for visualization/teleop
    bringup_pkg_share = FindPackageShare(package="bringup").find("bringup")
    mapping_file_path = os.path.join(bringup_pkg_share, "config/primeu_joint_mapping.json")
    primeu_joint_remap_node = Node(
        package="adam_retarget",
        executable="primeu_joint_remap.py",
        name="primeu_joint_remap",
        output="screen",
        parameters=[
            {
                "input_topic": "/adam/joint_states",
                "output_topic": "/primeu/remap_joint_states",
                "mapping_file": mapping_file_path,
            }
        ],
    )

    mocap_visual_joint_republisher_node = Node(
        package="primeu_bringup",
        executable="joint_state_visual_republisher.py",
        name="mocap_visual_joint_state_republisher",
        output="screen",
        parameters=[
            {
                "source_topics": [
                    "/primeu/remap_joint_states",
                    "/left_hand/joint_commands",
                    "/right_hand/joint_commands",
                ],
                "output_topic": "/primeu/mocap_visual_joint_states",
                "publish_rate": ParameterValue(
                    LaunchConfiguration("mocap_visual_rate"), value_type=float
                ),
                "source_stale_timeout_sec": 0.5,
                "left_hand_source_topic": "/left_hand/joint_commands",
                "right_hand_source_topic": "/right_hand/joint_commands",
                "default_joint_names": [
                    "waist_left_passive1_joint_z",
                    "waist_left_passive1_joint_y",
                    "waist_left_passive1_joint_x",
                    "waist_right_passive1_joint_z",
                    "waist_right_passive1_joint_y",
                    "waist_right_passive1_joint_x",
                    "neck_yaw_joint",
                    "neck_roll_joint",
                    "neck_pitch_joint",
                ],
                "default_joint_positions": [0.0] * 9,
            }
        ],
        condition=IfCondition(enable_mocap_visual),
    )

    head_pinocchio_ik_node = Node(
        package="adam_retarget",
        executable="head_pinocchio_ik.py",
        name="head_pinocchio_ik",
        output="screen",
        parameters=[
            {
                "joint_state_topic": "/joint_state_broadcaster/joint_states",
                "command_topic": "/primeu/control/human/raw/neck_commands",
                "visualization_source_topic": "",
                "visualization_joint_topic": "/primeu/mocap_visual_joint_states",
                "mocap_neck_frame": "noitom/Neck",
                "mocap_head_frame": "noitom/Head",
                "robot_base_frame": "chest_link",
                "robot_tip_frame": "neck_pitch_link",
                "publish_rate": ParameterValue(
                    LaunchConfiguration("head_ik_rate"), value_type=float
                ),
                "tf_timeout_sec": 0.05,
                "command_smoothing_alpha": 0.35,
                "auto_calibrate": True,
                "solver_max_nfev": 12,
                "publish_debug_topics": False,
            }
        ],
        condition=LaunchConfigurationEquals("enable_head_ik", "true"),
    )

    noitom_mocap = Node(
        package="noitom_mocap",
        executable="noitom_mocap",
        name="noitom_robot_tf_broadcaster",
        parameters=[
            {"root_frame": "world_noitom_yup", "child_prefix": "noitom/"}
        ],
    )

    ld.add_action(robot_state_publisher_node)
    ld.add_action(adam_retarget_node)
    ld.add_action(primeu_joint_remap_node)
    ld.add_action(mocap_visual_joint_republisher_node)
    ld.add_action(primeu_controller_bridge_node)
    ld.add_action(waist_retarget_bridge_node)
    ld.add_action(head_pinocchio_ik_node)
    ld.add_action(world_to_world_noitom)
    ld.add_action(noitom_yup_to_zup)
    ld.add_action(mocap_robot_anchor)
    ld.add_action(noitom_mocap)

    rviz_config_file = "rviz/robot.rviz"
    rviz_config_file_path = os.path.join(bringup_pkg_share, rviz_config_file)
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file_path],
        condition=LaunchConfigurationEquals("visual", "rviz2"),
    )
    ld.add_action(rviz_node)

    return ld
