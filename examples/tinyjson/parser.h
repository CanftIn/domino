#ifndef DOMINO_EXAMPLES_PARSER_H_
#define DOMINO_EXAMPLES_PARSER_H_

#include <cctype>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "json_ast.h"

namespace domino {

namespace examples {

namespace tinyjson {

class Parser {
 public:
  explicit Parser(const std::string& str) : json(str), Pos(0) {}

  std::unique_ptr<Node> parse() {
    skipWhitespace();
    switch (json[Pos]) {
      case 'n':
        return parseNull();
      case 't':
      case 'f':
        return parseBool();
      case '"':
        return parseString();
      case '[':
        return parseArray();
      case '{':
        return parseObject();
      default:
        return parseNumber();
    }
  }

 private:
  void skipWhitespace() {
    while (std::isspace(json[Pos])) {
      ++Pos;
    }
  }

  std::unique_ptr<Node> parseNull() {
    if (json.substr(Pos, 4) != "null") {
      throw std::runtime_error("invalid json");
    }
    Pos += 4;
    return std::make_unique<NullNode>();
  }

  std::unique_ptr<Node> parseBool() {
    if (json.substr(Pos, 4) == "true") {
      Pos += 4;
      return std::make_unique<BoolNode>(true);
    } else if (json.substr(Pos, 5) == "false") {
      Pos += 5;
      return std::make_unique<BoolNode>(false);
    } else {
      throw std::runtime_error("invalid json");
    }
  }

  std::unique_ptr<Node> parseString() {
    if (json[Pos] != '"') {
      throw std::runtime_error("invalid json");
    }
    ++Pos;
    std::string str;
    while (json[Pos] != '"') {
      if (json[Pos] == '\\') {
        ++Pos;
        switch (json[Pos]) {
          case '"':
            str += '"';
            break;
          case '\\':
            str += '\\';
            break;
          case '/':
            str += '/';
            break;
          case 'b':
            str += '\b';
            break;
          case 'f':
            str += '\f';
            break;
          case 'n':
            str += '\n';
            break;
          case 'r':
            str += '\r';
            break;
          case 't':
            str += '\t';
            break;
          case 'u': {
            ++Pos;
            std::stringstream ss;
            ss << std::hex << json.substr(Pos, 4);
            int code;
            ss >> code;
            str += static_cast<char>(code);
            Pos += 4;
            break;
          }
          default:
            throw std::runtime_error("invalid json");
        }
      } else {
        str += json[Pos];
      }
      ++Pos;
    }
    ++Pos;
    return std::make_unique<StringNode>(str);
  }

  std::unique_ptr<Node> parseArray() {
    ++Pos;
    Array arr;
    skipWhitespace();
    if (json[Pos] != ']') {
      while (true) {
        skipWhitespace();
        arr.push_back(parse());
        skipWhitespace();
        if (json[Pos] == ']') {
          break;
        }
        if (json[Pos] != ',') {
          throw std::runtime_error("invalid json");
        }
        ++Pos;
      }
    }
    ++Pos;
    return std::make_unique<ArrayNode>(std::move(arr));
  }

  std::unique_ptr<Node> parseObject() {
    ++Pos;
    Object obj;
    skipWhitespace();
    if (json[Pos] != '}') {
      while (true) {
        skipWhitespace();
        if (Pos >= json.size() || json[Pos] != '\"') {
          throw std::runtime_error("Expected a string key");
        }
        auto str = static_cast<StringNode*>(parseString().release());
        auto key = str->value();
        skipWhitespace();
        if (Pos >= json.size() || json[Pos] != ':') {
          throw std::runtime_error("Expected a colon");
        }
        ++Pos;
        obj[std::move(key)] = parse();
        skipWhitespace();
        if (json[Pos] == '}') {
          break;
        }
        if (json[Pos] != ',') {
          throw std::runtime_error("Expected a comma");
        }
        ++Pos;
      }
    }
    ++Pos;
    return std::make_unique<ObjectNode>(std::move(obj));
  }

  std::unique_ptr<Node> parseNumber() {
    std::string str;
    while (std::isdigit(json[Pos]) || json[Pos] == '-' || json[Pos] == '+' ||
           json[Pos] == '.' || json[Pos] == 'e' || json[Pos] == 'E') {
      str += json[Pos];
      ++Pos;
    }
    if (str.find('.') != std::string::npos ||
        str.find('e') != std::string::npos ||
        str.find('E') != std::string::npos) {
      return std::make_unique<NumberNode>(std::stod(str));
    } else {
      return std::make_unique<NumberNode>(std::stoll(str));
    }
  }

 private:
  const std::string& json;
  size_t Pos;
};

std::unique_ptr<Node> parse(const std::string& json) {
  return Parser(json).parse();
}

}  // namespace tinyjson

}  // namespace examples

}  // namespace domino

#endif  // DOMINO_EXAMPLES_PARSER_H_