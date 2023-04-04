set_project("domino")
set_languages("c11", "c++17")
set_warnings("allextra")

set_config("cc", "clang")
set_config("cxx", "clang++")
set_config("ld", "clang++")

add_rules("mode.debug", "mode.release")

add_requires("gtest")

add_includedirs("./src")
add_includedirs("./src/support")

target("domino")
    set_kind("static")
    add_files("./src/*.cc")
    add_files("./src/support/*.cc")

add_subdirs('test')
