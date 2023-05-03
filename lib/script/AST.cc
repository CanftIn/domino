#include <domino/script/AST.h>
#include <domino/support/TypeSwitch.h>
#include <domino/support/raw_ostream.h>
#include <domino/util/Twine.h>

using namespace domino::script;

struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

class ASTDumper {
 public:
  void dump(ModuleAST *node);

 private:
  void dump(const VarType &type);
  void dump(VarDeclExprAST *varDecl);
  void dump(ExprAST *expr);
  void dump(ExprASTList *exprList);
  void dump(NumberExprAST *num);
  void dump(LiteralExprAST *node);
  void dump(VariableExprAST *node);
  void dump(ReturnExprAST *node);
  void dump(BinaryExprAST *node);
  void dump(CallExprAST *node);
  void dump(PrintExprAST *node);
  void dump(PrototypeAST *node);
  void dump(FunctionAST *node);

  void indent() {
    for (int i = 0; i < curIndent; ++i) {
      domino::errs() << "  ";
    }
  }

  int curIndent = 0;
};

template <typename T>
static std::string loc(T *node) {
  const auto &loc = node->loc();
  return (domino::Twine("@") + *loc.file + ":" + domino::Twine(loc.line) + ":" +
          domino::Twine(loc.col))
      .str();
}

#define INDENT()            \
  Indent level_(curIndent); \
  indent();

void ASTDumper::dump(ExprAST *expr) {
  domino::TypeSwitch<ExprAST *>(expr)
      .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
            PrintExprAST, ReturnExprAST, VarDeclExprAST, VariableExprAST>(
          [&](auto *node) { this->dump(node); })
      .Default([&](ExprAST *) {
        INDENT();
        domino::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
      });
}

/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value.
void ASTDumper::dump(VarDeclExprAST *varDecl) {
  INDENT();
  domino::errs() << "VarDecl " << varDecl->getName();
  dump(varDecl->getType());
  domino::errs() << " " << loc(varDecl) << "\n";
  dump(varDecl->getInitVal());
}

/// A "block", or a list of expression
void ASTDumper::dump(ExprASTList *exprList) {
  INDENT();
  domino::errs() << "Block {\n";
  for (auto &expr : *exprList) dump(expr.get());
  indent();
  domino::errs() << "} // Block\n";
}

/// A literal number, just print the value.
void ASTDumper::dump(NumberExprAST *num) {
  INDENT();
  domino::errs() << num->getValue() << " " << loc(num) << "\n";
}

/// Helper to print recursively a literal. This handles nested array like:
///    [ [ 1, 2 ], [ 3, 4 ] ]
/// We print out such array with the dimensions spelled out at every level:
///    <2,2>[<2>[ 1, 2 ], <2>[ 3, 4 ] ]
void printLitHelper(ExprAST *litOrNum) {
  // Inside a literal expression we can have either a number or another literal
  if (auto *num = domino::dyn_cast<NumberExprAST>(litOrNum)) {
    domino::errs() << num->getValue();
    return;
  }
  auto *literal = domino::cast<LiteralExprAST>(litOrNum);

  // Print the dimension for this literal first
  domino::errs() << "<";
  domino::interleaveComma(literal->getDims(), domino::errs());
  domino::errs() << ">";

  // Now print the content, recursing on every element of the list
  domino::errs() << "[ ";
  domino::interleaveComma(literal->getValues(), domino::errs(),
                        [&](auto &elt) { printLitHelper(elt.get()); });
  domino::errs() << "]";
}

/// Print a literal, see the recursive helper above for the implementation.
void ASTDumper::dump(LiteralExprAST *node) {
  INDENT();
  domino::errs() << "Literal: ";
  printLitHelper(node);
  domino::errs() << " " << loc(node) << "\n";
}

/// Print a variable reference (just a name).
void ASTDumper::dump(VariableExprAST *node) {
  INDENT();
  domino::errs() << "var: " << node->getName() << " " << loc(node) << "\n";
}

/// Return statement print the return and its (optional) argument.
void ASTDumper::dump(ReturnExprAST *node) {
  INDENT();
  domino::errs() << "Return\n";
  if (node->getExpr().has_value()) return dump(*node->getExpr());
  {
    INDENT();
    domino::errs() << "(void)\n";
  }
}

/// Print a binary operation, first the operator, then recurse into LHS and RHS.
void ASTDumper::dump(BinaryExprAST *node) {
  INDENT();
  domino::errs() << "BinOp: " << node->getOp() << " " << loc(node) << "\n";
  dump(node->getLHS());
  dump(node->getRHS());
}

/// Print a call expression, first the callee name and the list of args by
/// recursing into each individual argument.
void ASTDumper::dump(CallExprAST *node) {
  INDENT();
  domino::errs() << "Call '" << node->getCallee() << "' [ " << loc(node) << "\n";
  for (auto &arg : node->getArgs()) dump(arg.get());
  indent();
  domino::errs() << "]\n";
}

/// Print a builtin print call, first the builtin name and then the argument.
void ASTDumper::dump(PrintExprAST *node) {
  INDENT();
  domino::errs() << "Print [ " << loc(node) << "\n";
  dump(node->getExpr());
  indent();
  domino::errs() << "]\n";
}

/// Print type: only the shape is printed in between '<' and '>'
void ASTDumper::dump(const VarType &type) {
  domino::errs() << "<";
  domino::interleaveComma(type.shape, domino::errs());
  domino::errs() << ">";
}

/// Print a function prototype, first the function name, and then the list of
/// parameters names.
void ASTDumper::dump(PrototypeAST *node) {
  INDENT();
  domino::errs() << "Proto '" << node->getName() << "' " << loc(node) << "\n";
  indent();
  domino::errs() << "Params: [";
  domino::interleaveComma(node->getArgs(), domino::errs(),
                        [](auto &arg) { domino::errs() << arg->getName(); });
  domino::errs() << "]\n";
}

/// Print a function, first the prototype and then the body.
void ASTDumper::dump(FunctionAST *node) {
  INDENT();
  domino::errs() << "Function \n";
  dump(node->getProto());
  dump(node->getBody());
}

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(ModuleAST *node) {
  INDENT();
  domino::errs() << "Module:\n";
  for (auto &f : *node) dump(&f);
}

namespace domino::script {

// Public API
void dump(ModuleAST &module) { ASTDumper().dump(&module); }

}  // namespace domino::script
