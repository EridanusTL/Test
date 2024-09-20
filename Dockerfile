FROM ubuntu:24.04

RUN apt update && apt upgrade -y 
RUN apt install -y git vim curl wget sudo build-essential cmake python-is-python3

# # ROS2
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg 
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null 
# RUN apt update && apt install ros-dev-tools -y
# RUN apt install ros-jazzy-desktop -y

