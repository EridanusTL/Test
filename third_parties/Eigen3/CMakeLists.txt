add_library(Eigen3 INTERFACE)
set_target_properties(
        Eigen3 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include/eigen3
)
add_library(Eigen3::Eigen ALIAS Eigen3)

# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/eigen3 DESTINATION include) 