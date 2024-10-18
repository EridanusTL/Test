#include <iostream>
#include <thread>

#include "Singleton.h"
#include "control.h"

Singleton* Singleton::instance = nullptr;
std::mutex Singleton::mtx;

void createSingletonInstance() {
  Singleton* singleton = Singleton::getInstance();
  std::cout << "1: " << singleton << std::endl;
}

int main(char* argc, char* argv[]) {
  if (0) {
    createSingletonInstance();
    Control control;
    control.DoSomething();
    return 0;
  } else {
    // 启动多个线程，几乎同时调用 getInstance()
    std::thread t1(createSingletonInstance);
    std::thread t2(createSingletonInstance);

    t1.join();
    t2.join();
  }
}
