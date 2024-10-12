#include <Eigen/Dense>
#include <iostream>

int main(int argc, char** argv) {
  Eigen::Vector3d kp(1, 2, 3);
  Eigen::Vector3d q(1, 1, 1);
  Eigen::Vector3d res = kp.cwiseProduct(q);
  std::cout << res << std::endl;
  return 0;
}