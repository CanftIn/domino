#ifndef DOMINO_EXAMPLES_JSON_AST_H_
#define DOMINO_EXAMPLES_JSON_AST_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace domino {

namespace examples {

namespace tinyjson {

enum class JsonType {
  kNull,
  kBool,
  kNumber,
  kString,
  kArray,
  kObject,
};

class Node {
 public:
  Node(JsonType type) : Type(type) {}

  virtual ~Node() = default;

  JsonType type() const { return Type; }

 private:
  JsonType Type;
};

using Object = std::map<std::string, std::unique_ptr<Node>>;
using Array = std::vector<std::unique_ptr<Node>>;

class NullNode : public Node {
 public:
  NullNode() : Node(JsonType::kNull) {}

  static bool classof(const Node* c) { return c->type() == JsonType::kNull; }
};

class BoolNode : public Node {
 public:
  BoolNode(bool value) : Node(JsonType::kBool), Val(value) {}

  bool value() const { return Val; }

  static bool classof(const Node* c) { return c->type() == JsonType::kBool; }

 private:
  bool Val;
};

class NumberNode : public Node {
 public:
  NumberNode(double value) : Node(JsonType::kNumber), Val(value) {}

  double value() const { return Val; }

  static bool classof(const Node* c) { return c->type() == JsonType::kNumber; }

 private:
  double Val;
};

class StringNode : public Node {
 public:
  StringNode(std::string value) : Node(JsonType::kString), Val(value) {}

  const std::string& value() const { return Val; }

  static bool classof(const Node* c) { return c->type() == JsonType::kString; }

 private:
  std::string Val;
};

class ArrayNode : public Node {
 public:
  ArrayNode(Array Arr) : Node(JsonType::kArray), Val(std::move(Arr)) {}

  const Array& value() { return Val; }

  static bool classof(const Node* c) { return c->type() == JsonType::kArray; }

 private:
  Array Val;
};

class ObjectNode : public Node {
 public:
  ObjectNode(Object Obj) : Node(JsonType::kObject), Val(std::move(Obj)) {}

  const Object& value() { return Val; }

  static bool classof(const Node* c) { return c->type() == JsonType::kObject; }

 private:
  Object Val;
};

std::unique_ptr<Node> parse(const std::string& json);

}  // namespace tinyjson

}  // namespace examples

}  // namespace domino

#endif  // DOMINO_EXAMPLES_JSON_AST_H_
