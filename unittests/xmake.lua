set_group("unittests")
set_default(get_config("unittests"))

add_requires("gtest")
add_packages("gtest")
add_links("gtest_main")

add_includedirs("../src")
add_deps("domino")

add_subdirs("http")
add_subdirs("util")
add_subdirs("support")
add_subdirs("meta")