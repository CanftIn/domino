#include <domino/script/Dialect.h>
#include <domino/script/MLIRCodeGen.h>
#include <domino/script/Parser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

using namespace domino::script;
using namespace script;

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<domino::script::ModuleAST> parseInputFile(llvm::StringRef buffer) {
  domino::script::LexerBuffer lexer(buffer.begin(), buffer.end(), std::string("null"));
  domino::script::Parser parser(lexer);
  return parser.parseModule();
}

int dumpMLIR(llvm::StringRef buffer) {
  mlir::MLIRContext context;
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<domino::script::ScriptDialect>();

  auto moduleAST = parseInputFile(buffer);
  if (!moduleAST) return 6;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      script::mlirGen(context, *moduleAST);
  if (!module) return 1;

  module->dump();
  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  //mlir::registerAsmPrinterCLOptions();
  //mlir::registerMLIRContextCLOptions();

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return 1;
  }

  std::string filename = argv[1];
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
    return 1;
  }

  std::string line;
  std::string buffer;
  while (std::getline(file, line)) {
    buffer += line + "\n";
  }
  std::cout << buffer << std::endl;

  dumpMLIR(buffer);

  return 0;
}
