cmake_minimum_required(VERSION 3.5)
project(TEST)

# file(GLOB_RECURSE CPP_FILES     
# ${PROJECT_SOURCE_DIR}/src/*.cc 
# ${PROJECT_SOURCE_DIR}/src/*.cpp
# )
set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "Release")
file(GLOB_RECURSE CPP_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
message("--------Proj Src Dir: " ${CPP_FILES})
include_directories(include)

# add_library(${PROJECT_NAME} SHARED ${CPP_FILES})

add_executable(main ${CPP_FILES})