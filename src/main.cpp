#include <iostream>
#include <Eigen/Dense>

int main(int argc, char **argv)
{
    std::cout << Eigen::Vector3d(1, 2, 3).dot(Eigen::Vector3d(1, 2, 3));

    return 0;
}