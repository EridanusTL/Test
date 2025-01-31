#include "singleton.h"

#include <iostream>
#include <thread>

void fun() {
  Singleton& singleton = Singleton::GetInstance();
  std::cout << &singleton << std::endl;
}

int main(int argc, char** argv) {
  std::thread t1(fun);
  std::thread t2(fun);

  t1.join();
  t2.join();
  return 0;
}