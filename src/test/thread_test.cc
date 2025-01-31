#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

std::mutex mtx1;
std::mutex mtx2;
std::timed_mutex time_mtx;

class SingleTon {
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

int fun() {
  int x = 0;

  for (size_t i = 0; i < 1000; i++) {
    x++;
  }
  return x;
}

int main(int argc, char* argv[]) {
  int x = 0;

  std::future<int> future_int = std::async(std::launch::async, fun);
  std::cout << fun() << std::endl;
  std::cout << future_int.get() << std::endl;

  std::cout << "Thread joined!" << std::endl;

  return 0;
}