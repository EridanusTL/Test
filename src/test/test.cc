#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif
#include <iostream>

#include <glog/logging.h>
#include <Eigen/Dense>

int main(int argc, char** argv) {
  Eigen::Matrix3d R_1 = Eigen::Matrix3d::Identity();
  std::cout << "Test!" << std::endl;

  return 0;
}