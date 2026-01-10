#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"

namespace pnd {
namespace noitom_mocap {

using NoitomMocapPubCb = std::function<void(std::vector<geometry_msgs::msg::TransformStamped>&)>;

class DataHandler {
 public:
  static DataHandler& getInstance() {
    static DataHandler instance;
    return instance;
  }
  ~DataHandler() {}
  DataHandler(DataHandler const&) = delete;
  DataHandler& operator=(DataHandler const&) = delete;

  void set_root_frame(std::string root_frame) { root_frame_ = std::move(root_frame); }
  void set_child_prefix(std::string child_prefix) { child_prefix_ = std::move(child_prefix); }
  const std::string& root_frame() const { return root_frame_; }
  const std::string& child_prefix() const { return child_prefix_; }

  void init();
  bool exit_flag() { return exit_flag_; }
  void reg_handle(NoitomMocapPubCb func) { handle_func_ = func; }
  void call() {
    if (handle_func_) {
      handle_func_(data_);
    }
  }
  void exit() {
    exit_flag_ = true;
    if (thread_ == nullptr) {
      return;
    }
    thread_->join();
    delete thread_;
    thread_ = nullptr;
  }

 public:
  std::vector<geometry_msgs::msg::TransformStamped> data_;
  std::map<std::string, int> joint_index_;

 private:
  DataHandler() = default;

 private:
  std::string root_frame_ = "world";
  std::string child_prefix_;

 private:
  std::thread* thread_;
  bool exit_flag_ = false;
  NoitomMocapPubCb handle_func_;
};

}  // namespace noitom_mocap
}  // namespace pnd