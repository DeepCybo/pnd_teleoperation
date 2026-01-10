
#include <memory>

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "noitom_mocap/noitom_data_handler.h"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"

class RobotTfBroadcaster : public rclcpp::Node {
 public:
  RobotTfBroadcaster() : Node("noitom_robot_tf_broadcaster") {
  this->declare_parameter<std::string>("root_frame", "world");
  this->declare_parameter<std::string>("child_prefix", "");

  const auto root_frame = this->get_parameter("root_frame").as_string();
  const auto child_prefix = this->get_parameter("child_prefix").as_string();
  pnd::noitom_mocap::DataHandler::getInstance().set_root_frame(root_frame);
  pnd::noitom_mocap::DataHandler::getInstance().set_child_prefix(child_prefix);

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    pnd::noitom_mocap::DataHandler::getInstance().reg_handle(
        std::bind(&RobotTfBroadcaster::broadcast_joint_transforms, this, std::placeholders::_1));

  // Start mocap thread after configuration.
  pnd::noitom_mocap::DataHandler::getInstance().init();
  }

 private:
  void broadcast_joint_transforms(std::vector<geometry_msgs::msg::TransformStamped>& data) {
    for (size_t idx = 0; idx < data.size(); idx++) {
      data[idx].header.stamp = this->now();
    }
    // auto t3 = this->now();
    // RCLCPP_INFO(this->get_logger(), "[this->now()] sec:%lf nano:%ld", t3.seconds(), t3.nanoseconds());
    tf_broadcaster_->sendTransform(std::move(data));
  }

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RobotTfBroadcaster>());
  rclcpp::shutdown();
  pnd::noitom_mocap::DataHandler::getInstance().exit();
  return 0;
}