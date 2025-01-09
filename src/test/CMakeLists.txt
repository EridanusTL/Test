project(test)

add_executable(hello_world hello_world.cc)
add_executable(glog_test glog_test.cc)
add_executable(my_math_test my_math_test.cc)
add_executable(thread_test thread_test.cc)

target_link_libraries(glog_test glog)
target_link_libraries(my_math_test Eigen3::Eigen glog)
target_link_libraries(thread_test Eigen3::Eigen glog pthread)