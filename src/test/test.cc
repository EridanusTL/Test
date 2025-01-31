// #include <fcntl.h>
// #include <linux/input.h>
// #include <stdio.h>
// #include <sys/stat.h>
// #include <sys/types.h>
// #include <unistd.h>

// int main() {
//   int keys_fd;
//   input_event t;

//   char device[] = "/dev/input/event5";

//   keys_fd = open(device, O_RDONLY);

//   if (keys_fd <= 0) {
//     printf("Open %s device error!\n", device);
//     return -1;
//   }

//   bool up = 0, down = 0, left = 0, right = 0;  // true for pressed, flase for released
//   double vel = 0, yaw = 0;
//   while (1) {
//     if (read(keys_fd, &t, sizeof(t)) != sizeof(t)) continue;

//     if (t.type != EV_KEY) continue;

//     if (t.code == KEY_ESC) break;

//     if (t.code == KEY_LEFT) left = t.value;
//     if (t.code == KEY_RIGHT) right = t.value;
//     if (t.code == KEY_UP) up = t.value;
//     if (t.code == KEY_DOWN) down = t.value;

//     yaw = left ? -1 : right ? 1 : 0;
//     vel = up ? 1 : down ? -1 : 0;

//     printf("Control vel %.1lf, yaw %.1lf\n", vel, yaw);
//   }

//   close(keys_fd);
//   return 0;
// }

#include <fcntl.h>
#include <linux/input.h>
#include <linux/types.h>
#include <unistd.h>
#include <iostream>

int main() {
  int fd = open("/dev/input/event5", O_RDONLY);  // 打开键盘设备文件
  if (fd == -1) {
    perror("Failed to open device file");
    return 1;
  }

  std::cout << "按下任意键退出程序，同时按下多个键会显示所有按键的代码。\n";

  struct input_event ev;
  while (true) {
    if (read(fd, &ev, sizeof(struct input_event)) == sizeof(struct input_event)) {
      //   if (ev.type == EV_KEY && ev.value == 1) {  // 检测按键按下事件
      std::cout << "按键 " << ev.code << " 被按下。\n";
      if (ev.code == KEY_ESC) {  // 如果按下 ESC 键，退出程序
        break;
      }
      //   }
    }
  }

  close(fd);
  return 0;
}