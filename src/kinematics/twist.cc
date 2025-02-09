#include "twist.h"

Eigen::AngleAxisd Twist::RotationMatrixToAxisAngle(const Eigen::Matrix3d& R) {
  Eigen::Vector3d w;
  double theta;
  if (R == Eigen::Matrix3d::Identity()) {  // case 1
    theta = 0;
    w = Eigen::Vector3d(0, 0, 0);
  } else if (R.trace() == -1) {  // case 2
    theta = M_PI;
    if (1 + R(2, 2) != 0) {
      w = 1 / std::sqrt(2 * (1 + R(2, 2))) * Eigen::Vector3d(R(0, 2), R(1, 2), 1 + R(2, 2));
    } else if (1 + R(1, 1) != 0) {
      w = 1 / std::sqrt(2 * (1 + R(1, 1))) * Eigen::Vector3d(R(0, 1), 1 + R(1, 1), R(2, 1));
    } else {
      w = 1 / std::sqrt(2 * (1 + R(0, 0))) * Eigen::Vector3d(1 + R(0, 0), R(1, 0), R(2, 0));
    }
  } else {  // case 3
    theta = std::acos((R.trace() - 1) / 2);
    Eigen::Matrix3d w_skew = (R - R.transpose()) / (2 * std::sin(theta));
    w = Eigen::Vector3d(w_skew(2, 1), w_skew(0, 2), w_skew(1, 0));
  }

  return Eigen::AngleAxisd(theta, w);
}

Eigen::Matrix3d Twist::AxisAngleToRotationMatrix(Eigen::AngleAxisd angle_axis) {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity() +
                      std::sin(angle_axis.angle()) * mymath::SkewMatrix(angle_axis.axis()) +
                      (1 - std::cos(angle_axis.angle())) * mymath::SkewMatrix(angle_axis.axis()) *
                          mymath::SkewMatrix(angle_axis.axis());

  return R;
}

Eigen::Matrix<double, 6, 6> Twist::AdjointRepresentationMatrix(const Eigen::Affine3d& T) {
  Eigen::Matrix3d R = T.rotation();
  Eigen::Vector3d p = T.translation();
  Eigen::Matrix<double, 6, 6> Adj = Eigen::Matrix<double, 6, 6>::Zero();
  Adj.block<3, 3>(0, 0) = R;
  Adj.block<3, 3>(3, 0) = mymath::SkewMatrix(p) * R;
  Adj.block<3, 3>(3, 3) = R;

  return Adj;
}

Eigen::Matrix<double, 6, 6> Twist::AdjointRepresentationMatrix(const Eigen::Matrix4d& T) {
  Eigen::Matrix3d R = T.block(0, 0, 3, 3);
  Eigen::Vector3d p = T.block(0, 3, 3, 1);
  Eigen::Matrix<double, 6, 6> Adj = Eigen::Matrix<double, 6, 6>::Zero();
  Adj.block<3, 3>(0, 0) = R;
  Adj.block<3, 3>(3, 0) = mymath::SkewMatrix(p) * R;
  Adj.block<3, 3>(3, 3) = R;

  return Adj;
}

Eigen::Vector<double, 6> Twist::TransformMatrixToScrewAxis(const Eigen::Affine3d& T) {
  Eigen::Vector3d w;
  Eigen::Vector3d v;
  double theta;
  Eigen::Matrix3d R = T.rotation();
  Eigen::AngleAxisd angle_axis = RotationMatrixToAxisAngle(R);
  Eigen::Matrix3d G_inv = Eigen::Matrix3d::Identity() / theta - mymath::SkewMatrix(w) / 2 +
                          (1 / theta - 1 / 2 / std::tan(theta / 2)) * mymath::SkewMatrix(w) * mymath::SkewMatrix(w);
  v = G_inv * T.translation();
  Eigen::Vector<double, 6> S;
  S << w, v;
  return S;
}

Eigen::Vector<double, 6> Twist::TransformMatrixToScrewAxis(const Eigen::Matrix4d& T) {
  Eigen::Vector3d w;
  Eigen::Vector3d v;
  double theta;
  Eigen::Matrix3d R = T.block(0, 0, 3, 3);
  Eigen::Vector3d p = T.block(0, 3, 3, 1);
  Eigen::Vector<double, 6> S = Eigen::Vector<double, 6>::Zero();

  if (R == Eigen::Matrix3d::Identity()) {
    w = Eigen::Vector3d(0, 0, 0);
    v = p / p.norm();
    theta = p.norm();
  } else {
    RotationMatrixToAxisAngle(R);
    Eigen::Matrix3d G = theta * Eigen::Matrix3d::Identity() + (1 - std::cos(theta)) * mymath::SkewMatrix(w) +
                        (theta - std::sin(theta)) * mymath::SkewMatrix(w) * mymath::SkewMatrix(w);

    Eigen::Matrix3d G_inv = Eigen::Matrix3d::Identity() / theta - mymath::SkewMatrix(w) / 2 +
                            (1 / theta - 1.0 / 2 / std::tan(theta / 2)) * mymath::SkewMatrix(w) * mymath::SkewMatrix(w);

    v = G_inv * p;
  }

  S << w, v;
  return S;
}

Eigen::Matrix4d Twist::ScrewAxisToTransformMatrix(const Eigen::Vector<double, 6>& S, double theta) {
  Eigen::Vector3d w = S.head(3);
  Eigen::Vector3d v = S.tail(3);
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  if (w == Eigen::Vector3d::Zero()) {
    T.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
    T.block(0, 3, 3, 1) = theta * v;
  } else {
    Eigen::Matrix3d G = theta * Eigen::Matrix3d::Identity() + (1 - std::cos(theta)) * mymath::SkewMatrix(w) +
                        (theta - std::sin(theta)) * mymath::SkewMatrix(w) * mymath::SkewMatrix(w);

    T.block(0, 0, 3, 3) = AxisAngleToRotationMatrix(Eigen::AngleAxisd(theta, w));
    T.block(0, 3, 3, 1) = G * v;
  }

  return T;
}

Twist::Twist() { LOG(INFO) << "Twist constructor called."; }

Twist::~Twist() {}

int main(int argc, char** argv) {
  google::InitGoogleLogging("glog_test");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  Eigen::Vector<double, 6> S_2(0, 1, 0, -0.089, 0, 0);
  double theta_2 = -M_PI / 2;

  Eigen::Vector<double, 6> S_5(0, 0, -1, -0.109, 0.817, 0);
  double theta_5 = M_PI / 2;

  Twist twist;
  Eigen::Matrix4d T1 = twist.ScrewAxisToTransformMatrix(S_2, theta_2);
  Eigen::Matrix4d T2 = twist.ScrewAxisToTransformMatrix(S_5, theta_5);
  LOG(INFO) << "T2:\n" << T2;

  return 0;
}
