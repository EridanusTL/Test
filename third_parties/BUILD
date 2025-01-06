cc_library(
    name = "eigen",
    srcs = [],
    hdrs = glob(["Eigen/**"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rbdl",
    srcs = [
        "rbdl/lib/librbdl.so.3.3.1",
        "rbdl/lib/librbdl_urdfreader.so.3.3.1",
    ],
    hdrs = glob(["rbdl/include/**"]),
    copts = ["-std=c++17"],
    includes = [
        "rbdl/include",
        "rbdl/inclued/rbdl",
        "rbdl/inclued/rbdl/addons/urdfreader",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mujoco-lib",
    srcs = ["mujoco/lib/libmujoco.so.3.2.2"],
    hdrs = glob(["mujoco/**"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "qpOASES",
    srcs = [
        "qpOASES/lib/libqpOASES.a",
    ],
    hdrs = glob(["qpOASES/**/*"]),
    includes = [
        "qpOASES/include",
        "qpOASES/include/qpOASES",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "glog",
    srcs = [
        "glog/lib/libglog.so",
        "glog/lib/libglog.so.0.7.1",
        "glog/lib/libglog.so.2",
    ],
    hdrs = glob(["glog/include/glog/*.h"]),
    includes = ["glog/include"],
    visibility = ["//visibility:public"],
)
