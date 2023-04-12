cmake_minimum_required(VERSION 3.6)

project(domino_unittests LANGUAGES CXX)

find_program(CMAKE_MEMORYCHECK_COMMAND valgrind)
set(memcheck_command ${CMAKE_MEMORYCHECK_COMMAND} ${CMAKE_MEMORYCHECK_COMMAND_OPTIONS} --error-exitcode=1 --leak-check=full)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

enable_testing()
find_package(GTest REQUIRED)

add_subdirectory(http)
add_subdirectory(support)
add_subdirectory(util)
add_subdirectory(meta)