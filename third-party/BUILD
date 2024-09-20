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
