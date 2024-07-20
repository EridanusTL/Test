#include <glog/logging.h>
#include <iostream>

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  //   FLAGS_alsologtostderr = 1;
  //   google::SetLogSeverityLevel(google::GLOG_INFO);

  std::cout << "Hello World!" << std::endl;

  LOG(INFO) << "This is an info message";
  LOG(WARNING) << "This is a warning message";
  LOG(ERROR) << "This is an error message";

  //   google::ShutdownGoogleLogging();
  return 0;
}
