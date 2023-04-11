cmake_minimum_required(VERSION 3.6)
project(
  domino
  VERSION 0.0.1
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Debug")

set(project_version "${${PROJECT_NAME}_VERSION}")


message(STATUS "Project '${PROJECT_NAME}', version: '${project_version}'")

option(DOMINO_OPT_BUILD_UNITTESTS "Build all domino unittests" ON)

# CMake helpers:
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

include_directories(src)
# Create the domino C++ library
set(domino_SRCS
  src/domino/http/uri_parser.cc
)

set(STATIC_LIB_NAME ${PROJECT_NAME})
set(SHARED_LIB_NAME ${PROJECT_NAME}-shared)

add_library(${STATIC_LIB_NAME} STATIC ${domino_SRCS})
add_library(${SHARED_LIB_NAME} SHARED ${domino_SRCS})

if (DOMINO_OPT_BUILD_UNITTESTS)
  add_subdirectory(unittests)
endif()