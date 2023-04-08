add_packages("gtest")
add_packages("gtest")
add_links("gtest_main")

add_deps("domino")

target("uri_parser_test")
  set_kind("binary")
  add_files("uri_parser_test.cc")

target("type_traits_test")
  set_kind("binary")
  add_files("type_traits_test.cc")

target("small_vector_test")
  set_kind("binary")
  add_files("small_vector_test.cc")