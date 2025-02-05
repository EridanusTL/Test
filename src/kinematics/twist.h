#ifndef TWIST_H
#define TWIST_H

#include <Eigen/Dense>

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <glog/logging.h>

class Twist {
 public:
  Eigen::Vector3d RotationMatrixToAxisAngle(const Eigen::Matrix3d& R);

  Twist();
  ~Twist();

 private:
};

#endif