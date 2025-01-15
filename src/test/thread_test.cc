#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

std::mutex mtx1;
std::mutex mtx2;
std::timed_mutex time_mtx;

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

void fun(int& x) {
  for (size_t i = 0; i < 2; i++) {
    std::unique_lock<std::timed_mutex> lock(time_mtx, std::defer_lock);
    if (lock.try_lock_for(std::chrono::seconds(1))) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      x++;
    }
  }
}

int main(int argc, char* argv[]) {
  int x = 0;

  std::thread t1(&fun, std::ref(x));
  std::thread t2(&fun, std::ref(x));

  t1.join();
  t2.join();
  std::cout << "x: " << x << std::endl;
  std::cout << "Thread joined!" << std::endl;

  return 0;
}