#include <Eigen/Dense>
#include <iostream>

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <glog/logging.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging("eigen_test");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  Eigen::MatrixXd M, N;
  M = Eigen::MatrixXd::Random(5, 5);
  N.row(0) = M.row(0);
  LOG(INFO) << "Matrix M:\n" << M;
  LOG(INFO) << "Matrix N:\n" << N;
  M = Eigen::MatrixXd::Random(3, 3);
  LOG(INFO) << "Matrix M:\n" << M;
  return 0;
}