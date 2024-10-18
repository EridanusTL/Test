FROM ubuntu:22.04

ENV LANG=en_US.UTF-8
ENV TZ=Asia/Shanghai

RUN apt update && apt upgrade -y && \
    apt install -y git vim curl wget sudo build-essential cmake python-is-python3 python3-pip

# # ROS2
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg 
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null 
# RUN apt update && apt install ros-dev-tools -y
# RUN apt install ros-jazzy-desktop -y


# Install ROS2 Humble from deb package
RUN apt install -y software-properties-common && \
    add-apt-repository universe && \
    apt update && apt install curl -y && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt update && apt upgrade -y 

# apt install ros-humble-desktop -y && \
# apt install ros-dev-tools -y && \
# RUN pip install 

