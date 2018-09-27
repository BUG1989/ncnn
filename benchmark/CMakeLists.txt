
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

add_executable(benchncnn benchncnn.cpp)
set_property(TARGET benchncnn PROPERTY COMPILE_FLAGS "-fpie")
set_property(TARGET benchncnn PROPERTY LINK_FLAGS "-pie")
target_link_libraries(benchncnn ncnn)

add_executable(benchncnn-int8 benchncnn-int8.cpp)
set_property(TARGET benchncnn-int8 PROPERTY COMPILE_FLAGS "-fpie")
set_property(TARGET benchncnn-int8 PROPERTY LINK_FLAGS "-pie")
target_link_libraries(benchncnn-int8 ncnn)
