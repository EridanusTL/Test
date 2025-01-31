#ifndef JOYSTICK_H
#define JOYSTICK_H

#include <iostream>
#include <memory>

#include <termios.h>
#include <unistd.h>

class Joystick {
 public:
  char getch();
  double& forward() { return forward_; }
  double& turn() { return turn_; }
  void Damping(double& delta, double& value);
  void Run(std::shared_ptr<Joystick> joystick);

  static std::shared_ptr<Joystick> GetIntance() { return instance_; }

 private:
  double forward_;
  double turn_;

  static std::shared_ptr<Joystick> instance_;
  Joystick();
};
std::shared_ptr<Joystick> Joystick::instance_ = std::shared_ptr<Joystick>(new Joystick());

#endif