#include "control.h"

void Control::DoSomething() {
  Singleton* singleton = Singleton::getInstance();
  std::cout << "2: " << singleton << std::endl;
}