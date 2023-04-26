#ifndef DOMINO_SCRIPT_AST_H_
#define DOMINO_SCRIPT_AST_H_

#include <domino/script/Lexer.h>
#include <domino/support/Casting.h>
#include <domino/util/ArrayRef.h>
#include <domino/util/StringRef.h>

#include <optional>
#include <utility>
#include <vector>

namespace domino {

namespace script {

class ExprAST {
 public:
  virtual ~ExprAST() = default;

  virtual void dump() const = 0;
};

class NumberExprAST : public ExprAST {
 public:
  NumberExprAST(double value) : value(value) {}

  void dump() const override;

 private:
  double value;
};

class VariableExprAST : public ExprAST {
 public:
  VariableExprAST(StringRef name) : name(name) {}

  void dump() const override;

 private:
  StringRef name;
};

class BinaryExprAST : public ExprAST {
 public:
  BinaryExprAST(char op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
      : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  void dump() const override;

 private:
  char op;
  std::unique_ptr<ExprAST> lhs;
  std::unique_ptr<ExprAST> rhs;
};

}

}

#endif  // DOMINO_SCRIPT_AST_H_