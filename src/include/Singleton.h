#ifndef Singleton_H_
#define Singleton_H_

#include <iostream>
#include <mutex>

class Singleton {
 private:
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;

  static Singleton* instance;

  // 静态互斥锁，用于线程安全的加锁操作
  static std::mutex mtx;

  Singleton() { std::cout << "Constructor called!" << std::endl; };

 public:
  static Singleton* getInstance() {
    // 第一次检测：如果实例已经存在，直接返回，避免不必要的加锁
    if (instance == nullptr) {
      std::lock_guard<std::mutex> lock(mtx);  // 加锁保护，确保线程安全
      // 第二次检测：加锁后再次检查实例是否存在，以防止多个线程在第一次检测后都通过判断
      if (instance == nullptr) {
        instance = new Singleton();  // 如果实例仍然不存在，创建新的实例
      }
    }
    return instance;  // 返回唯一的实例
  }

  // 单例类中的其他方法
  void doSomething() { std::cout << "Doing something..." << std::endl; }
};

#endif
