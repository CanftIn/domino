set_group("examples")
set_default(get_config("examples"))

add_includedirs("../include")
add_includedirs("./tinyjson")

add_deps("domino")

folds = {"tinyjson"}

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
  if has_config("memcheck") then
    on_run(function(target)
      local argv = {}
      table.insert(argv, target:targetfile())
      table.insert(argv, "--leak-check=full")
      os.execv("valgrind", argv)
    end)
  end
end
