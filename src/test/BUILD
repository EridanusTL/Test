cc_binary(
    name = "urdf_reader_test",
    srcs = ["urdf_reader_test.cc"],
    deps = [
        "//third_party:eigen",
        "//third_party:rbdl",
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
    deps = [
        "//src/include:eigen",
    ],
)

cc_binary(
    name = "singleton_test",
    srcs = ["singleton_test.cc"],
    deps = [
        "//src/include:control",
        "//src/include:singleton",
    ],
)

cc_binary(
    name = "qpOASES_test",
    srcs = ["qpOASES_test.cc"],
    deps = [
        "//third_party:eigen",
        "//third_party:glog",
        "//third_party:qpOASES",
    ],
)


cc_binary(
    name = "hello-world",
    srcs = ["hello-world.cc"],
)