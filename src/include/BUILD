cc_library(
    name = "singleton",
    hdrs = ["Singleton.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "control",
    srcs = ["control.cc"],
    hdrs = ["control.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [":singleton"],
)
