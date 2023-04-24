set_project("domino")
set_version("0.0.1")

option("domino_src",  {description = "workflow src", default = "$(projectdir)/src"})
option("unittests",   {description = "build unittests", default = true})
option("memcheck",      {description = "valgrind memcheck", default = false})

if is_mode("release") then
    set_optimize("faster")
    set_strip("all")
elseif is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
end

set_languages("c11", "c++17")
set_warnings("allextra")

set_config("cc", "clang")
set_config("cxx", "clang++")
set_config("ld", "clang++")

add_cflags("-fPIC", "-pipe")
add_cxxflags("-fPIC", "-pipe", "-Wno-invalid-offsetof")

includes("**/xmake.lua")

add_includedirs("include")

target("http")
    set_kind("object")
    add_files("lib/http/*.cc")

target("util")
    set_kind("object")
    add_files("lib/util/*.cc")

target("domino")
    set_kind("$(kind)")
    add_deps("http")
    add_deps("util")