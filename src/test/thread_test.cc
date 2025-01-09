#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

std::mutex mtx1;
std::mutex mtx2;

class A {
 public:
  void fun1() {
    mtx1.lock();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    mtx2.lock();
    mtx1.unlock();
    mtx2.unlock();
  }

  void fun2() {
    mtx1.lock();
    mtx2.lock();
    mtx1.unlock();
    mtx2.unlock();
  }
};

int main(int argc, char* argv[]) {
  std::shared_ptr<A> a = std::make_shared<A>();

  std::thread t1(&A::fun1, a);
  std::thread t2(&A::fun2, a);

  t1.join();
  t2.join();
  std::cout << "Thread joined!" << std::endl;

  return 0;
}