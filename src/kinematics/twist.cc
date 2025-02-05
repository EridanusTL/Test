#include "twist.h"

Eigen::Vector3d Twist::RotationMatrixToAxisAngle(const Eigen::Matrix3d& R) { return Eigen::Vector3d(); }

Twist::Twist() {}

Twist::~Twist() {}

int main(int argc, char** argv) {
  google::InitGoogleLogging("glog_test");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  LOG(INFO) << "Hello World!";
  return 0;
}