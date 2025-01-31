#include "joystick.h"
#include <fcntl.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

Joystick::Joystick() {
  forward_ = 0;
  turn_ = 0;
}

char Joystick::getch() {
  char buf = 0;
  struct termios old = {0};

  // 获取当前终端设置
  if (tcgetattr(0, &old) < 0) {
    perror("tcgetattr()");
    return 0;
  }

  struct termios newt = old;
  newt.c_lflag &= ~ICANON;  // 关闭规范模式
  newt.c_lflag &= ~ECHO;    // 关闭回显
  newt.c_cc[VMIN] = 1;      // 至少读取一个字符
  newt.c_cc[VTIME] = 0;     // 不设超时

  // 设置终端为非阻塞模式
  if (tcsetattr(0, TCSANOW, &newt) < 0) {
    perror("tcsetattr ICANON");
    return 0;
  }

  // 设置终端为非阻塞输入
  fcntl(0, F_SETFL, O_NONBLOCK);

  // 读取一个字符
  if (read(0, &buf, 1) < 0) {
    // 如果没有输入，不阻塞，返回0
    return 0;
  }
  old.c_lflag |= ICANON;
  // 恢复回显
  old.c_lflag |= ECHO;

  // 恢复终端设置
  tcsetattr(0, TCSADRAIN, &old);
  return buf;
}

void Joystick::Damping(double& delta, double& value) {
  if (delta > 0) {
    value -= 0.01;
    std::abs(value) < 0.01 ? value = 0 : value = value;
  } else if (delta < 0) {
    value += 0.01;
    std::abs(value) < 0.01 ? value = 0 : value = value;
  } else {
    value = 0;
  }
}

void Joystick::Run(std::shared_ptr<Joystick> joystick) {
  double delta_forward, delta_turn;

  while (true) {
    auto start = std::chrono::steady_clock::now();
    delta_forward = joystick->forward() - 0;
    delta_turn = joystick->turn() - 0;

    system("clear");
    std::cout << "Use w, s, a, d, or q." << std::endl;
    std::cout << "Forward: " << std::fixed << std::setprecision(3) << joystick->forward() << std::endl;
    std::cout << "Turn: " << std::fixed << std::setprecision(3) << joystick->turn() << std::flush;

    char c = joystick->getch();
    switch (c) {
      case 'w':
        // Move forward
        joystick->forward() += 0.01;
        joystick->forward() = std::clamp(joystick->forward(), -1.0, 1.0);
        joystick->Damping(delta_turn, joystick->turn());

        break;
      case 's':
        // Move backward
        joystick->forward() -= 0.01;
        joystick->forward() = std::clamp(joystick->forward(), -1.0, 1.0);
        joystick->Damping(delta_turn, joystick->turn());

        break;
      case 'a':
        // Move left
        joystick->turn() += 0.01;
        joystick->turn() = std::clamp(joystick->turn(), -1.0, 1.0);
        joystick->Damping(delta_forward, joystick->forward());

        break;
      case 'd':
        // Move right
        joystick->turn() -= 0.01;
        joystick->turn() = std::clamp(joystick->turn(), -1.0, 1.0);
        joystick->Damping(delta_forward, joystick->forward());

        break;
      case 'q':
        // Quit
        std::cout << "Joystick Quit." << std::endl;
        return;
      default:

        joystick->Damping(delta_forward, joystick->forward());
        joystick->Damping(delta_turn, joystick->turn());

        break;
    }

    auto end = std::chrono::steady_clock::now();

    auto time_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::this_thread::sleep_for(std::chrono::nanoseconds(30000000) - time_elapsed);
  }
}

int main(int argc, char* argv[]) {
  std::shared_ptr<Joystick> joystick = Joystick::GetIntance();
  std::thread joystickThread = std::thread(&Joystick::Run, joystick, joystick);
  joystickThread.join();
  return 0;
}