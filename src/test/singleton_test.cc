#include <iostream>

#include "Singleton.h"
#include "control.h"

Singleton* Singleton::instance = nullptr;
std::mutex Singleton::mtx;

int main(char* argc, char* argv[]) {
  Singleton* singleton = Singleton::getInstance();
  std::cout << "1: " << singleton << std::endl;

  Control control;
  control.DoSomething();
  return 0;
}