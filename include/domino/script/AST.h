#ifndef DOMINO_SCRIPT_AST_H_
#define DOMINO_SCRIPT_AST_H_

#include <domino/script/Lexer.h>
#include <domino/support/Casting.h>
#include <domino/util/ArrayRef.h>
#include <domino/util/StringRef.h>

#include <cstdint>
#include <memory>
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
 public:
  ReturnExprAST(Location location, std::optional<std::unique_ptr<ExprAST>> expr)
      : ExprAST(Expr_Return, std::move(location)), expr(std::move(expr)) {}

  std::optional<ExprAST*> getExpr() {
    if (expr.has_value()) return expr->get();
    return std::nullopt;
  }

  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Return; }

 private:
  std::optional<std::unique_ptr<ExprAST>> expr;
};

class BinaryExprAST : public ExprAST {
 public:
  BinaryExprAST(Location location, char op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
      : ExprAST(Expr_BinOp, std::move(location)),
        op(op),
        lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  char getOp() const { return op; }
  ExprAST* getLHS() const { return lhs.get(); }
  ExprAST* getRHS() const { return rhs.get(); }

  static bool classof(const ExprAST* c) { return c->getKind() == Expr_BinOp; }

 private:
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;
};

class CallExprAST : public ExprAST {
 public:
  CallExprAST(Location loc, const std::string& callee,
              std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(Expr_Call, std::move(loc)),
        callee(callee),
        args(std::move(args)) {}

  StringRef getCallee() const { return callee; }
  ArrayRef<std::unique_ptr<ExprAST>> getArgs() const { return args; }

  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Call; }

 private:
  std::string callee;
  std::vector<std::unique_ptr<ExprAST>> args;
};

class PrintExprAST : public ExprAST {
 public:
  PrintExprAST(Location location, std::unique_ptr<ExprAST> expr)
      : ExprAST(Expr_Print, std::move(location)), expr(std::move(expr)) {}

  ExprAST* getExpr() const { return expr.get(); }

  static bool classof(const ExprAST* c) { return c->getKind() == Expr_Print; }

 private:
  std::unique_ptr<ExprAST> expr;
};

class PrototypeAST {
 public:
  PrototypeAST(Location location, StringRef name,
               std::vector<std::unique_ptr<VariableExprAST>> args)
      : location(std::move(location)), name(name), args(std::move(args)) {}

  const Location& loc() { return location; }
  StringRef getName() const { return name; }
  ArrayRef<std::unique_ptr<VariableExprAST>> getArgs() const { return args; }

 private:
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VariableExprAST>> args;
};

class FunctionAST {
 public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprASTList> body)
      : proto(std::move(proto)), body(std::move(body)) {}

  PrototypeAST* getProto() const { return proto.get(); }
  ExprASTList* getBody() const { return body.get(); }

 private:
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprASTList> body;
};

class ModuleAST {
 public:
  ModuleAST(std::vector<FunctionAST> functions)
      : functions(std::move(functions)) {}

  auto begin() { return functions.begin(); }
  auto end() { return functions.end(); }

 private:
  std::vector<FunctionAST> functions;
};

void dump(ModuleAST&);

}  // namespace script

}  // namespace domino

#endif  // DOMINO_SCRIPT_AST_H_