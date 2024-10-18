cc_binary(
    name = "urdf_reader_test",
    srcs = ["urdf_reader_test.cc"],
    copts = ["-std=c++17"],
    deps = [
        "//third-party:eigen",
        "//third-party:rbdl",
    ],
)

cc_binary(
    name = "glog_test",
    srcs = ["glog_test.cc"],
    deps = ["@com_github_google_glog//:glog"],
)

cc_binary(
    name = "eigen_test",
    srcs = ["eigen_test.cc"],
    copts = ["-std=c++17"],
    deps = [
        "//src/include:control",
        "//src/include:singleton",
    ],
)

cc_binary(
    name = "singleton_test",
    srcs = ["singleton_test.cc"],
    copts = ["-std=c++17"],
    deps = [
        "//src/include:control",
        "//src/include:singleton",
    ],
)
