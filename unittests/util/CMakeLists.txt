cmake_minimum_required(VERSION 3.6)

set(UNITTESTS_LIST
  "small_vector_test"
  "to_string_test"
  "logging_test"
  "stringref_test"
)

foreach(src ${UNITTESTS_LIST})
  message(STATUS "unittest files found: ${src}.cc")
  add_executable(${src} ${src}.cc)
  target_link_libraries(${src} GTest::gtest GTest::gtest_main domino)
  add_test(${src} ${src})
  add_dependencies(check ${src})
endforeach()

foreach(src ${UNITTESTS_FILES})
  add_test(${src}-memory-check ${memcheck_command} ./${src})
endforeach()