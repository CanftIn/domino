cmake_minimum_required(VERSION 3.6)

set(tinyjson_SRCS
  tinyjson.example.cc
  )

set(FILE_NAME tinyjson.example)

message(STATUS "build example: ${FILE_NAME}")
add_executable(${FILE_NAME} ${tinyjson_SRCS})
target_link_libraries(${FILE_NAME} domino)
add_test(${FILE_NAME} ${FILE_NAME})
add_dependencies(check ${FILE_NAME})
add_test(${FILE_NAME}-memory-check ${memcheck_command} ./${FILE_NAME})