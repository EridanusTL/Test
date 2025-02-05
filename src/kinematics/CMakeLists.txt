project(kinematics)

add_executable(twist twist.cc)
target_link_libraries(twist glog Eigen3::Eigen)