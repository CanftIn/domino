cmake_minimum_required(VERSION 3.6)

file(GLOB UNITTESTS_LIST *.cc)

foreach(FILE_PATH ${UNITTESTS_LIST})
  STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_NAME ${FILE_PATH})
  message(STATUS "unittest files found: ${FILE_NAME}")
  add_executable(${FILE_NAME} ${FILE_NAME}.cc)
  target_link_libraries(${FILE_NAME} GTest::gtest GTest::gtest_main domino)
  add_test(${FILE_NAME} ${FILE_NAME})
  add_dependencies(check ${FILE_NAME})
  add_test(${FILE_NAME}-memory-check ${memcheck_command} ./${FILE_NAME})
endforeach()