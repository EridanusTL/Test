#ifndef TWIST_H
#define TWIST_H

#include "my_math.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <glog/logging.h>

class Twist {
 public:
  Eigen::AngleAxisd RotationMatrixToAxisAngle(const Eigen::Matrix3d& R);
  Eigen::Matrix3d AxisAngleToRotationMatrix(Eigen::AngleAxisd angle_axis);
  Eigen::Matrix<double, 6, 6> AdjointRepresentationMatrix(const Eigen::Affine3d& T);
  Eigen::Matrix<double, 6, 6> AdjointRepresentationMatrix(const Eigen::Matrix4d& T);
  Eigen::Vector<double, 6> TransformMatrixToScrewAxis(const Eigen::Affine3d& T);
  Eigen::Vector<double, 6> TransformMatrixToScrewAxis(const Eigen::Matrix4d& T);
  Eigen::Matrix4d ScrewAxisToTransformMatrix(const Eigen::Vector<double, 6>& S, double theta);

  Twist();
  ~Twist();

 private:
};

#endif