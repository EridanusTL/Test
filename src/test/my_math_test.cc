#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif
#include <iostream>

#include <glog/logging.h>
#include <Eigen/Dense>
namespace mymath {
Eigen::Matrix3d EuleToRotation(const double& roll, const double& pitch, const double& yaw) {
  Eigen::AngleAxisd rot_z(yaw, Eigen::Vector3d::UnitZ());
  Eigen::AngleAxisd rot_y(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rot_x(roll, Eigen::Vector3d::UnitX());

  Eigen::Matrix3d R = (rot_z * rot_y * rot_x).toRotationMatrix();

  return R;
}

Eigen::Quaterniond EulerToQuaternion(double roll, double pitch, double yaw) {
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

// Eigen::Vector3d RotationToEuler(Eigen::Matrix3d R) {
//   Eigen::Vector3d euler = R.eulerAngles(2, 1, 0);
//   double tmp = euler(0);
//   euler(0) = euler(2);
//   euler(2) = tmp;
//   return euler;
// }

Eigen::Vector3d RotationToEuler(Eigen::Matrix3d R) {
  double pitch = std::atan2(-R(2, 0), std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0)));
  double roll = std::atan2(R(1, 0) / cos(pitch), R(0, 0) / cos(pitch));
  double yaw = std::atan2(R(2, 1) / cos(pitch), R(2, 2) / cos(pitch));
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
}  // namespace mymath
int main(int argc, char** argv) {
  google::InitGoogleLogging("glog_test");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  Eigen::Matrix3d R_2 = mymath::EuleToRotation(M_PI / 4, -M_PI / 4, M_PI / 2);
  LOG(WARNING) << "R_2: \n" << R_2;

  Eigen::Vector3d euler = mymath::RotationToEuler(R_2);
  LOG(WARNING) << "Euler: \n" << euler;

  Eigen::Quaterniond q = mymath::EulerToQuaternion(M_PI / 4, -M_PI / 4, M_PI / 2);
  LOG(WARNING) << "Quaternion: \n" << q.w() << " " << q.x() << " " << q.y() << " " << q.z();

  Eigen::Vector3d euler_2 = mymath::QuaternionToEuler(q);
  LOG(WARNING) << "Euler_2: \n" << euler_2;

  Eigen::Matrix3d R_3 = mymath::QuaternionToRotationMatrix(q);
  LOG(WARNING) << "R_3: \n" << R_3;

  Eigen::Quaterniond q_2 = mymath::RotationMatrixToQuaternion(R_3);
  LOG(WARNING) << "Quaternion_2: \n" << q_2.w() << " " << q_2.x() << " " << q_2.y() << " " << q_2.z();

  Eigen::Matrix3d R_4 = Eigen::Matrix3d::Identity();
  Eigen::Vector3d euler_3 = mymath::RotationToEuler(R_4);
  LOG(INFO) << "Euler: \n" << euler_3;

  return 0;
}