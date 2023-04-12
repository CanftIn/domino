folds = {"http", "util", "support", "meta"}

function all_tests(folds)
  local res = {}
  for fold in pairs(folds) do 
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
  print(test[1])
  print(test[2])
end