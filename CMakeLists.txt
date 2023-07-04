cmake_minimum_required(VERSION 3.6)
project(
  domino
  VERSION 0.0.1
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE "Debug")
set(project_version "${${PROJECT_NAME}_VERSION}")

message(STATUS "Project '${PROJECT_NAME}', version: '${project_version}'")

option(DOMINO_OPT_BUILD_UNITTESTS "Build all domino unittests" ON)
option(DOMINO_OPT_BUILD_BENCHMARK "Build all domino benchmark" OFF)
option(DOMINO_OPT_BUILD_EXAMPLES "Build all domino examples" ON)
option(DOMINO_OPT_BUILD_UNITTESTS_SCRIPT "Build domino script" OFF)

# temp define: https://discourse.llvm.org/t/python-api-problem/945
add_compile_options(-fno-rtti)

# CMake helpers:
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

include(CTest)

include_directories(include)

# Create the domino C++ library
set(domino_SRCS
  lib/http/UriParser.cc
  lib/support/Error.cc
  lib/support/FormatVariadic.cc
  lib/support/NativeFormatting.cc
  lib/support/Process.cc
  lib/support/Program.cc
  lib/support/raw_ostream.cc
  lib/support/hash/MD5.cc
  lib/support/hash/SHA1.cc
  lib/support/hash/SHA256.cc
  lib/support/filesystem/Errno.cc
  lib/support/filesystem/Path.cc
  lib/util/StringRef.cc
  lib/util/Twine.cc
  lib/util/Hashing.cc
  lib/util/StringExtras.cc
  lib/rpc/concurrency/Mutex.cc
  lib/rpc/transport/Socket.cc
)

set(STATIC_LIB_NAME ${PROJECT_NAME})
set(SHARED_LIB_NAME ${PROJECT_NAME}-shared)

if (DOMINO_OPT_BUILD_UNITTESTS_SCRIPT)
  set(MLIR_DIR /usr/local/lib/cmake/mlir)

  find_package(LLVM REQUIRED CONFIG)
  find_package(MLIR REQUIRED CONFIG)

  list(APPEND domino_SRCS
    lib/script/AST.cc
    lib/script/Dialect.cc
    lib/script/MLIRCodeGen.cc
  )
  add_library(${STATIC_LIB_NAME} STATIC ${domino_SRCS})
  target_link_libraries(${STATIC_LIB_NAME}
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms
    LLVMSupport
    LLVMOption
  )
  #add_library(${SHARED_LIB_NAME} SHARED ${domino_SRCS})
else()
  add_library(${STATIC_LIB_NAME} STATIC ${domino_SRCS})
  #add_library(${SHARED_LIB_NAME} SHARED ${domino_SRCS})
endif()

if (DOMINO_OPT_BUILD_UNITTESTS)
  add_subdirectory(unittests #[[EXCLUDE_FROM_ALL]])
endif()

if (DOMINO_OPT_BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

if (DOMINO_OPT_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()