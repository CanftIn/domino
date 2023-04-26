#include <iostream>

#include "parser.h"

int main() {
  std::string json_str =
      R"({"key": "value", "array": [1, 2, 3], "nested": {"foo": "bar"},
       "number": 42.0, "bool": true, "null": null})";

  try {
    auto root = domino::examples::tinyjson::parse(json_str);
    std::cout << "JSON parsed successfully" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  return 0;
}