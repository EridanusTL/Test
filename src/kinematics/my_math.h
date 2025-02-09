#ifndef MY_MATH_H
#define MY_MATH_H

#include <iostream>

#include <Eigen/Dense>
namespace mymath {
Eigen::Matrix3d EulerToRotationMatrix(const double& yaw, const double& pitch, const double& roll) {
  Eigen::AngleAxisd rot_z(yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd rot_y(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rot_x(roll, Eigen::Vector3d::UnitX());

  Eigen::Matrix3d R = (rot_z * rot_y * rot_x).toRotationMatrix();

  return R;
}

Eigen::Quaterniond EulerToQuaternion(const double& yaw, const double& pitch, const double& roll) {
  double cr = cos(roll / 2);
  double sr = sin(roll / 2);
  double cp = cos(pitch / 2);
  double sp = sin(pitch / 2);
  double cy = cos(yaw / 2);
  double sy = sin(yaw / 2);

  double w = cr * cp * cy + sr * sp * sy;
  double x = sr * cp * cy - cr * sp * sy;
  double y = cr * sp * cy + sr * cp * sy;
  double z = cr * cp * sy - sr * sp * cy;

  return Eigen::Quaterniond(w, x, y, z);
}

Eigen::Quaterniond EulerToQuaternion(const Eigen::Vector3d& eluer) {
  double roll = eluer[2];
  double pitch = eluer[1];
  double yaw = eluer[0];

  double cr = cos(roll / 2);
  double sr = sin(roll / 2);
  double cp = cos(pitch / 2);
  double sp = sin(pitch / 2);
  double cy = cos(yaw / 2);
  double sy = sin(yaw / 2);

  double w = cr * cp * cy + sr * sp * sy;
  double x = sr * cp * cy - cr * sp * sy;
  double y = cr * sp * cy + sr * cp * sy;
  double z = cr * cp * sy - sr * sp * cy;

  return Eigen::Quaterniond(w, x, y, z);
}

Eigen::Matrix3d RPYToRotationMatrix(const double& roll, const double& pitch, const double& yaw) {
  Eigen::AngleAxisd rot_z(yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd rot_y(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rot_x(roll, Eigen::Vector3d::UnitX());

  Eigen::Matrix3d R = (rot_z * rot_y * rot_x).toRotationMatrix();

  return R;
}

Eigen::Quaterniond RPYToQuaternion(const double& roll, const double& pitch, const double& yaw) {
  double cr = cos(roll / 2);
  double sr = sin(roll / 2);
  double cp = cos(pitch / 2);
  double sp = sin(pitch / 2);
  double cy = cos(yaw / 2);
  double sy = sin(yaw / 2);

  double w = cr * cp * cy + sr * sp * sy;
  double x = sr * cp * cy - cr * sp * sy;
  double y = cr * sp * cy + sr * cp * sy;
  double z = cr * cp * sy - sr * sp * cy;

  return Eigen::Quaterniond(w, x, y, z);
}

Eigen::Quaterniond RPYToQuaternion(Eigen::Vector3d& RPY) {
  double roll = RPY[0];
  double pitch = RPY[1];
  double yaw = RPY[2];

  double cr = cos(roll / 2);
  double sr = sin(roll / 2);
  double cp = cos(pitch / 2);
  double sp = sin(pitch / 2);
  double cy = cos(yaw / 2);
  double sy = sin(yaw / 2);

  double w = cr * cp * cy + sr * sp * sy;
  double x = sr * cp * cy - cr * sp * sy;
  double y = cr * sp * cy + sr * cp * sy;
  double z = cr * cp * sy - sr * sp * cy;

  return Eigen::Quaterniond(w, x, y, z);
}

Eigen::Vector3d RotationMatrixToEuler(Eigen::Matrix3d& R) {
  double pitch = std::atan2(-R(2, 0), std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0)));
  double yaw = std::atan2(R(1, 0) / cos(pitch), R(0, 0) / cos(pitch));
  double roll = std::atan2(R(2, 1) / cos(pitch), R(2, 2) / cos(pitch));
  return Eigen::Vector3d(yaw, pitch, roll);
}

Eigen::Vector3d RotationMatrixToRPY(Eigen::Matrix3d& R) {
  double pitch = std::atan2(-R(2, 0), std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0)));
  double yaw = std::atan2(R(1, 0) / cos(pitch), R(0, 0) / cos(pitch));
  double roll = std::atan2(R(2, 1) / cos(pitch), R(2, 2) / cos(pitch));
  return Eigen::Vector3d(roll, pitch, yaw);
}

Eigen::Quaterniond RotationMatrixToQuaternion(Eigen::Matrix3d& R) {
  Eigen::Quaterniond quat(R);
  return quat;
}

Eigen::Vector3d QuaternionToEuler(Eigen::Quaterniond& q) {
  double yaw = std::atan2(2 * (q.w() * q.z() + q.x() * q.y()), 1 - 2 * (q.y() * q.y() + q.z() * q.z()));   // Z轴旋转
  double pitch = std::asin(-2 * (q.x() * q.z() - q.w() * q.y()));                                          // Y轴旋转
  double roll = std::atan2(2 * (q.w() * q.x() + q.y() * q.z()), 1 - 2 * (q.x() * q.x() + q.y() * q.y()));  // X轴旋转

  return Eigen::Vector3d(yaw, pitch, roll);
}

Eigen::Vector3d QuaternionToRPY(Eigen::Quaterniond& q) {
  double yaw = std::atan2(2 * (q.w() * q.z() + q.x() * q.y()), 1 - 2 * (q.y() * q.y() + q.z() * q.z()));   // Z轴旋转
  double pitch = std::asin(-2 * (q.x() * q.z() - q.w() * q.y()));                                          // Y轴旋转
  double roll = std::atan2(2 * (q.w() * q.x() + q.y() * q.z()), 1 - 2 * (q.x() * q.x() + q.y() * q.y()));  // X轴旋转

  return Eigen::Vector3d(roll, pitch, yaw);
}

Eigen::Matrix3d QuaternionToRotationMatrix(Eigen::Quaterniond& q) {
  Eigen::Matrix3d R;
  R << 1 - 2 * (q.y() * q.y() + q.z() * q.z()), 2 * (q.x() * q.y() - q.z() * q.w()),
      2 * (q.x() * q.z() + q.y() * q.w()), 2 * (q.x() * q.y() + q.z() * q.w()), 1 - 2 * (q.x() * q.x() + q.z() * q.z()),
      2 * (q.y() * q.z() - q.x() * q.w()), 2 * (q.x() * q.z() - q.y() * q.w()), 2 * (q.y() * q.z() + q.x() * q.w()),
      1 - 2 * (q.x() * q.x() + q.y() * q.y());

  return R;
}

Eigen::Matrix3d SkewMatrix(Eigen::Vector3d vec) {
  Eigen::Matrix3d skew{{0, -vec[2], vec[1]}, {vec[2], 0, -vec[0]}, {-vec[1], vec[0], 0}};
  return skew;
}
}  // namespace mymath

#endif