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

    # URDF (PrimeU)
    package_name = "primeu_description"
    urdf_name = "urdf/primeu_robot.urdf"
    urdf_pkg_share = FindPackageShare(package=package_name).find(package_name)
    urdf_model_path = os.path.join(urdf_pkg_share, urdf_name)

    with open(urdf_model_path, "r") as infp:
        robot_desc = infp.read()

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_desc}],
        remappings=[("/joint_states", "/primeu/joint_states")],
    )

    # transform noitom y-up to z-up (match existing ADAM noitom pipeline)
    static_transform_publisher_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="screen",
        arguments=[
            "0",
            "0",
            "0",
            "0.707106781",
            "0",
            "0",
            "0.707106781",
            "world_zup",
            "world",
        ],
        remappings=[("/tf", "/mocap/tf"), ("/tf_static", "/mocap/tf_static")],
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
                "base_frame": "world_zup",
                "control_loop_rate": 100.0,
                "config_json_path": default_config_json_path,
                "warm_start_trig_timeout": 0.2,
                "warm_start_duration": 5.0,
                "warm_start_slowdown_ratio": 0.1,
            },
        ],
        remappings=[
            ("/tf", "/mocap/tf"),
            ("/tf_static", "/mocap/tf_static"),
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
                "output_topic": "/primeu/joint_states",
                "mapping_file": mapping_file_path,
            }
        ],
    )

    noitom_mocap = Node(
        package="noitom_mocap",
        executable="noitom_mocap",
        name="noitom_robot_tf_broadcaster",
        remappings=[("/tf", "/mocap/tf")],
    )

    ld.add_action(robot_state_publisher_node)
    ld.add_action(adam_retarget_node)
    ld.add_action(primeu_joint_remap_node)
    ld.add_action(static_transform_publisher_node)
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
