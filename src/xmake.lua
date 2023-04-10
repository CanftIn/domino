includes("**/xmake.lua")

target("domino")
    set_kind("$(kind)")
    add_deps("http", "support", "util")