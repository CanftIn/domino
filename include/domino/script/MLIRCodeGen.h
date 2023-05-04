#ifndef DOMINO_SCRIPT_MLIR_CODEGEN_H_
#define DOMINO_SCRIPT_MLIR_CODEGEN_H_

#include <memory>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

namespace domino {

class MLIRContext;

template <typename OpTy>
class OwningOpRef;

class ModuleOp;

}  // namespace domino

namespace script {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
::mlir::OwningOpRef<::mlir::ModuleOp> mlirGen(::mlir::MLIRContext &context,
                                          ModuleAST &moduleAST);
}  // namespace script

#endif  // DOMINO_SCRIPT_MLIR_CODEGEN_H_