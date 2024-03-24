project(transformer)

find_package(Threads REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)


add_executable(attention attention_tb.cpp attention.cpp)
target_link_libraries(attention PRIVATE Threads::Threads)
