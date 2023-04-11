includes("**/xmake.lua")

add_includedirs(".")

target("domino")
    set_kind("$(kind)")
    add_deps("http", "support", "util")