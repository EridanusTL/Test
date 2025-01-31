#pragma once

#include <memory>
#include <mutex>

// class Singleton {
//  private:
//   Singleton() = default;

//   static std::shared_ptr<Singleton> instance_;
//   static std::mutex mtx_;

//  public:
//   Singleton(const Singleton&) = delete;
//   Singleton& operator=(const Singleton&) = delete;

//   static std::shared_ptr<Singleton> GetInstance() {
//     if (instance_ == nullptr) {
//       std::lock_guard<std::mutex> lock(mtx_);
//       if (instance_ == nullptr) {
//         instance_ = std::shared_ptr<Singleton>(new Singleton());
//       }
//     }
//     return instance_;
//   }
// };
// std::shared_ptr<Singleton> Singleton::instance_ = nullptr;
// std::mutex Singleton::mtx_;

class Singleton {
 private:
  Singleton() {};

  static Singleton instance_;

 public:
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;

  static Singleton& GetInstance() { return instance_; }
};

Singleton Singleton::instance_;