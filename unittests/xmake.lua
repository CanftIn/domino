set_group("unittests")
set_default(get_config("unittests"))

add_requires("gtest")
add_packages("gtest")

add_includedirs("../include")
add_deps("domino")

folds = {"http", "util", "support", "meta"}

function all_tests(folds)
  local res = {}
  for _, fold in pairs(folds) do 
    for _, x in ipairs(os.files(fold .. "/**.cc")) do
      local item = {}
      local s = path.filename(x)
      table.insert(item, s:sub(1, #s - 3)) -- target
      table.insert(item, path.relative(x, ".")) -- source
      table.insert(res, item)
    end
  end
  return res
end

for _, test in ipairs(all_tests(folds)) do
  target(test[1])
  set_kind("binary")
  add_files(test[2])
  add_links("gtest_main")
  if has_config("memcheck") then
    on_run(function(target)
      local argv = {}
      table.insert(argv, target:targetfile())
      table.insert(argv, "--leak-check=full")
      os.execv("valgrind", argv)
    end)
  end
end

-- if get_config("unittest_script") then
--   for _, test in ipairs(all_tests({"script"})) do
--     target(test[1])
--     set_kind("binary")
--     add_files(test[2])
--     if has_config("memcheck") then
--       on_run(function(target)
--         local argv = {}
--         table.insert(argv, target:targetfile())
--         table.insert(argv, "--leak-check=full")
--         os.execv("valgrind", argv)
--       end)
--     end
--   end
-- end