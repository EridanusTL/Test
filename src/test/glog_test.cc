#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <glog/logging.h>
#include <iostream>

int main(int argc, char* argv[]) {
  google::InitGoogleLogging("glog_test");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  std::cout << "Hello World!" << std::endl;

  LOG(INFO) << argv[0];
  LOG(INFO) << "This is an info message";
  LOG(WARNING) << "This is a warning message";
  LOG(ERROR) << "This is an error message";

  //   google::ShutdownGoogleLogging();
  return 0;
}
