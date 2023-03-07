add_packages("gtest")

target("uri_parser_test")
  add_deps("domino")
  add_files("uri_parser_test.cc")