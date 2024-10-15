load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = ["https://github.com/gflags/gflags/archive/v2.2.2.tar.gz"],
)

http_archive(
    name = "com_github_google_glog",
    sha256 = "c17d85c03ad9630006ef32c7be7c65656aba2e7e2fbfc82226b7e680c771fc88",
    strip_prefix = "glog-0.7.1",
    urls = ["https://github.com/google/glog/archive/v0.7.1.zip"],
)

# Eigen
http_archive(
    name = "eigen",
    build_file = "//:eigen.BUILD",
    sha256 = "3a66f9bfce85aff39bc255d5a341f87336ec6f5911e8d816dd4a3fdc500f8acf",
    strip_prefix = "eigen-eigen-c5e90d9e764e",
    url = "https://bitbucket.org/eigen/eigen/get/c5e90d9.tar.gz",
)

load("//tools:environ.bzl", "environment_repository")

environment_repository(
    name = "ros2_example_bazel_installed_environ",
    envvars = ["ROS2_DISTRO_PREFIX"],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",  # noqa
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",  # noqa
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",  # noqa
    ],
)

http_archive(
    name = "rules_python",
    sha256 = "0a8003b044294d7840ac7d9d73eef05d6ceb682d7516781a4ec62eeb34702578",  # noqa
    strip_prefix = "rules_python-0.24.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.24.0/rules_python-0.24.0.tar.gz",  # noqa
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

local_repository(
    name = "bazel_ros2_rules",
    path = "bazel_ros2_rules",
)

load("@bazel_ros2_rules//deps:defs.bzl", "add_bazel_ros2_rules_dependencies")

add_bazel_ros2_rules_dependencies()

load(
    "@bazel_ros2_rules//ros2:defs.bzl",
    "ros2_local_repository",
)
load(
    "@ros2_example_bazel_installed_environ//:environ.bzl",
    "ROS2_DISTRO_PREFIX",
)

# Please keep this list sorted
ROS2_PACKAGES = [
    "plotjuggler",
    "plotjuggler_ros",
    "action_msgs",
    "builtin_interfaces",
    "console_bridge_vendor",
    "rclcpp",
    "rclcpp_action",
    "rclpy",
    "ros2cli",
    "ros2cli_common_extensions",
    "rosbag2",
    "rosidl_default_generators",
    "tf2_py",
] + [
    # These are possible RMW implementations. Uncomment one and only one to
    # change implementations
    "rmw_cyclonedds_cpp",
    # "rmw_fastrtps_cpp",
]

RESOLVED_PREFIX = (
    ROS2_DISTRO_PREFIX if ROS2_DISTRO_PREFIX else "/opt/ros/humble"
)

ros2_local_repository(
    name = "ros2",
    include_packages = ROS2_PACKAGES,
    workspaces = [RESOLVED_PREFIX],
)
