add_library(glog SHARED IMPORTED GLOBAL)
set_target_properties(
        glog
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/libglog.so.0.7.1
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libglog.so.0.7.1 ${CMAKE_CURRENT_SOURCE_DIR}/lib/libglog.so.2
        DESTINATION lib)