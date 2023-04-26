#ifndef DOMINO_SCRIPT_AST_H_
#define DOMINO_SCRIPT_AST_H_

#include <domino/script/Lexer.h>
#include <domino/support/Casting.h>
#include <domino/util/ArrayRef.h>
#include <domino/util/StringRef.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace domino {

namespace script {

struct VarType {
  std::vector<int64_t> shape;
};

class ExprAST {
 public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(std::move(location)) {}

  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location& loc() { return location; }

 private:
  const ExprASTKind kind;
  Location location;
};

using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

class NumberExprAST : public ExprAST {
 public:
  NumberExprAST(Location location, double value)
      : ExprAST(Expr_Num, std::move(location)), value(value) {}

  double getValue() const { return value; }

  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Num; }

 private:
  double value;
};

class LiteralExprAST : public ExprAST {
 public:
  LiteralExprAST(Location location,
                 std::vector<std::unique_ptr<ExprAST>> values,
                 std::vector<int64_t> dims)
      : ExprAST(Expr_Literal, std::move(location)),
        values(std::move(values)),
        dims(std::move(dims)) {}

  ArrayRef<std::unique_ptr<ExprAST>> getValues() const { return values; }
  ArrayRef<int64_t> getDims() const { return dims; }

  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Literal; }

 private:
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;
};

class VariableExprAST : public ExprAST {
 public:
  VariableExprAST(Location location, StringRef name)
      : ExprAST(Expr_Var, std::move(location)), name(name) {}

  StringRef getName() const { return name; }

  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Var; }

 private:
  std::string name;
};

class VarDeclExprAST : public ExprAST {
 public:
  VarDeclExprAST(Location location, StringRef name, VarType type,
                 std::unique_ptr<ExprAST> initVal)
      : ExprAST(Expr_VarDecl, std::move(location)),
        name(name),
        type(type),
        initVal(std::move(initVal)) {}

  StringRef getName() const { return name; }
  ExprAST* getInitVal() const { return initVal.get(); }
  const VarType& getType() const { return type; }

  static bool classof(const ExprAST* c) { return c->getKind() == Expr_VarDecl; }

 private:
  std::string name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;
};

class ReturnExprAST : public ExprAST {
  

 private:
  
};

}  // namespace script

}  // namespace domino

#endif  // DOMINO_SCRIPT_AST_H_