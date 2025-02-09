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

  // 定义旋转矩阵（3x3）
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix = Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ());

  // 定义平移向量（3x1）
  Eigen::Vector3d translation_vector(1.0, 2.0, 3.0);

  // 使用Affine3d初始化齐次变换矩阵（4x4）
  Eigen::Affine3d transform(Eigen::Translation3d(translation_vector) * rotation_matrix);

  LOG(INFO) << "Transform matrix:\n" << transform.matrix();
  LOG(INFO) << "Rotation matrix:\n" << transform.rotation();
  LOG(INFO) << "Translation vector:\n" << transform.translation();

  Eigen::Vector3d point(1.0, 1.0, 1.0);
  Eigen::Vector3d transformed_point = transform * point;
  LOG(INFO) << "Original point: " << point.transpose();
  LOG(INFO) << "Transformed point: " << transformed_point.transpose();
  return 0;
}