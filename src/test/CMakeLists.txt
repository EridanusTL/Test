project(test)

add_executable(hello_world hello_world.cc)
add_executable(glog_test glog_test.cc)
add_executable(test test.cc)

target_link_libraries(glog_test glog)
target_link_libraries(test Eigen3::Eigen glog)