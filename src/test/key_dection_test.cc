#include <termios.h>
#include <unistd.h>
#include <iostream>

/**
 * @brief 从标准输入读取一个字符，不回显且立即返回
 *
 * 该函数通过修改终端属性，使得从标准输入读取字符时不回显，并且立即返回。
 * 读取完成后，恢复终端的原始属性。
 *
 * @return char 读取到的字符
 */
char getch() {
  // 用于存储读取到的字符
  char buf = 0;
  // 用于存储终端的原始属性
  struct termios old = {0};
  // 获取当前终端的属性
  if (tcgetattr(0, &old) < 0) perror("tcsetattr()");
  // 关闭规范模式，使得输入字符立即返回
  old.c_lflag &= ~ICANON;
  // 关闭回显，使得输入字符不显示在屏幕上
  old.c_lflag &= ~ECHO;
  // 设置最少读取一个字符
  old.c_cc[VMIN] = 1;
  // 设置读取超时时间为0，立即返回
  old.c_cc[VTIME] = 0;
  // 设置终端属性
  if (tcsetattr(0, TCSANOW, &old) < 0) perror("tcsetattr ICANON");
  // 从标准输入读取一个字符
  if (read(0, &buf, 1) < 0) perror("read()");
  // 恢复规范模式
  old.c_lflag |= ICANON;
  // 恢复回显
  old.c_lflag |= ECHO;
  // 恢复终端属性
  if (tcsetattr(0, TCSADRAIN, &old) < 0) perror("tcsetattr ~ICANON");
  // 返回读取到的字符
  return buf;
}

int main() {
  char ch;
  while (true) {
    ch = getch();  // 获取按键
    std::cout << "Key Pressed: " << ch << " ASCII: " << (int)ch << std::endl;
    if (ch == 27) break;  // 按ESC退出
  }
  return 0;
}