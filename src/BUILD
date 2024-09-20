cc_binary(
    name = "glog_test",
    srcs = ["glog_test.cc"],
    deps = ["@com_github_google_glog//:glog"],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        "//third-party:eigen",
        "//third-party:rbdl",
    ],
)

cc_binary(
    name = "urdf_reader_test",
    srcs = ["urdf_reader_test.cc"],
    copts = ["-std=c++17"],
    deps = [
        "//third-party:eigen",
        "//third-party:rbdl",
    ],
)
