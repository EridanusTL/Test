cmake_minimum_required(VERSION 3.5)
project(TEST)

# file(GLOB_RECURSE CPP_FILES     
# ${PROJECT_SOURCE_DIR}/src/*.cc 
# ${PROJECT_SOURCE_DIR}/src/*.cpp
# )

set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "Release")
include_directories(include)

# # Set opencv
# set(OpenCV_DIR ./thirdparties/opencv-4.9.0/build)
# find_package(OpenCV REQUIRED)
# include_directories(thirdparties/opencv-4.9.0/build/install/include)

add_executable(main src/main.cc)
target_link_libraries(main ${OpenCV_LIBS})


# message
message("--------OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
message("--------Proj Src Dir: " ${CPP_FILES})
message("--------OpenCV_LIBS: " ${OpenCV_LIBS})

