import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ld = LaunchDescription()

    visual_declare = DeclareLaunchArgument(
        "visual",
        default_value="foxglove",
        description="Choose visualization tool: rviz2 or foxglove",
        choices=["rviz2", "foxglove"],
    )
    ld.add_entity(visual_declare)
    enable_head_ik_declare = DeclareLaunchArgument(
        "enable_head_ik",
        default_value="true",
        description="Enable Noitom head tracking -> neck servo Pinocchio IK node",
    )
    ld.add_entity(enable_head_ik_declare)

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
        remappings=[("/joint_states", "/primeu/joint_states")],
    )

    # 3. Controller Bridge (Connects mocap JointState to ros2_control commands)
    #
    # NOTE: the waist output from this bridge is remapped to a dead topic so
    # the real /waist_servo_controller/commands is only driven by the
    # parallel IK pipeline:
    #     /primeu/remap_joint_states (roll/pitch/yaw)
    #         -> waist_retarget_bridge
    #         -> /waist_parallel_ik_node/target_rpy
    #         -> waist_parallel_ik_node
    #         -> /waist_servo_controller/commands
    # Letting the legacy bridge also publish to the real waist controller
    # would race against the IK node on the same topic.
    # (We cannot pass `waist_joints: []` here: ROS 2 Jazzy's launch rejects
    #  empty-list parameter values as an untyped tuple.)
    primeu_controller_bridge_node = Node(
        package="adam_retarget",
        executable="primeu_bridge_node.py",
        name="primeu_controller_bridge",
        output="screen",
        parameters=[
            {
                # High-rate publish for CSP mode EtherCAT drives
                "input_topic": "/primeu/remap_joint_states",
                "publish_rate": 500.0,
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
                "/waist_servo_controller/commands",
                "/waist_servo_controller/commands_disabled_by_parallel_ik",
            ),
        ],
    )

    # Waist retarget bridge: taps the retargeted joint state, extracts
    # waist_roll_passive_joint / waist_pitch_passive_joint / waist_yaw_joint,
    # clamps / sign-flips them, and publishes a Vector3(rad) to the waist
    # parallel-link IK node.
    #
    # Smoothing (OneEuro) is intentionally disabled here: the IK node is
    # now the rate master and runs its own OneEuro at the solve tick
    # (default 500 Hz). Double-filtering here would only add lag without
    # reducing jitter. Publish rate on this side only needs to be >= the
    # retarget source rate (typ. 100 Hz); the IK node interpolates up to
    # 500 Hz on its own tick.
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
                "publish_rate": 200.0,
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
                "control_loop_rate": 100.0,
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

    head_pinocchio_ik_node = Node(
        package="adam_retarget",
        executable="head_pinocchio_ik.py",
        name="head_pinocchio_ik",
        output="screen",
        parameters=[
            {
                "joint_state_topic": "/joint_states",
                "command_topic": "/neck_servo_controller/commands",
                "visualization_source_topic": "/primeu/remap_joint_states",
                "visualization_joint_topic": "/primeu/joint_states",
                "mocap_neck_frame": "noitom/Neck",
                "mocap_head_frame": "noitom/Head",
                "robot_base_frame": "chest_link",
                "robot_tip_frame": "neck_pitch_link",
                "publish_rate": 100.0,
                "tf_timeout_sec": 0.05,
                "command_smoothing_alpha": 0.35,
                "auto_calibrate": True,
                "publish_debug_topics": True,
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
