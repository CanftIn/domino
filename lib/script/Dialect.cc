#include <domino/script/Dialect.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace domino::script;

MLIR_DEFINE_EXPLICIT_TYPE_ID(::domino::script::ScriptDialect)
namespace domino {
namespace script {

ScriptDialect::ScriptDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
                      ::mlir::TypeID::get<::domino::script::ScriptDialect>()) {
  initialize();
}

ScriptDialect::~ScriptDialect() = default;

void ScriptDialect::initialize() {
  addOperations<::domino::script::AddOp, ::domino::script::ConstantOp,
                ::domino::script::FuncOp, ::domino::script::GenericCallOp,
                ::domino::script::MulOp, ::domino::script::PrintOp,
                ::domino::script::ReshapeOp, ::domino::script::ReturnOp,
                ::domino::script::TransposeOp>();
}

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = type.dyn_cast<FunctionType>()) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}

/// Verifier for the constant operation. This corresponds to the
/// `let hasVerifier = 1` in the op definition.
mlir::LogicalResult ConstantOp::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType = getResult().getType().dyn_cast<mlir::RankedTensorType>();
  if (!resultType) return success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  auto attrType = getValue().getType().cast<mlir::RankedTensorType>();
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError(
               "return type must match the one of the attached value "
               "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand()) return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || inputType.isa<mlir::UnrankedTensorType>() ||
      resultType.isa<mlir::UnrankedTensorType>())
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

mlir::LogicalResult TransposeOp::verify() {
  auto inputType = getOperand().getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();
  if (!inputType || !resultType) return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  resultType.getShape().rbegin())) {
    return emitError()
           << "expected result shape to be a transpose of the input";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_Ops0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) {
          return (elementType.isF64());
        }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind)
           << " #" << valueIndex
           << " must be tensor of 64-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_Ops1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::RankedTensorType>())) &&
         ((type.cast<::mlir::ShapedType>().hasStaticShape()))) &&
        ([](::mlir::Type elementType) {
          return (elementType.isF64());
        }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
                                      << " must be statically shaped tensor of "
                                         "64-bit float values, but got "
                                      << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::DenseFPElementsAttr>() &&
                 attr.cast<::mlir::DenseElementsAttr>()
                     .getType()
                     .getElementType()
                     .isF64()))) {
    return op->emitOpError("attribute '")
           << attrName
           << "' failed to satisfy constraint: 64-bit float elements attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '")
           << attrName << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) &&
                ((attr.cast<::mlir::TypeAttr>()
                      .getValue()
                      .isa<::mlir::FunctionType>())) &&
                ((attr.cast<::mlir::TypeAttr>()
                      .getValue()
                      .isa<::mlir::FunctionType>())))) {
    return op->emitOpError("attribute '")
           << attrName
           << "' failed to satisfy constraint: type attribute of function type";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) &&
                (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(),
                                [&](::mlir::Attribute attr) {
                                  return attr &&
                                         ((attr.isa<::mlir::DictionaryAttr>()));
                                })))) {
    return op->emitOpError("attribute '")
           << attrName
           << "' failed to satisfy constraint: Array of dictionary attributes";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::FlatSymbolRefAttr>()))) {
    return op->emitOpError("attribute '")
           << attrName
           << "' failed to satisfy constraint: flat symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_Ops0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #")
           << regionIndex
           << (regionName.empty() ? " " : " ('" + regionName + "') ")
           << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// ::domino::script::AddOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AddOpGenericAdaptorBase::AddOpGenericAdaptorBase(::mlir::DictionaryAttr attrs,
                                                 ::mlir::RegionRange regions)
    : odsAttrs(attrs), odsRegions(regions) {
  if (odsAttrs) odsOpName.emplace("script.add", odsAttrs.getContext());
}

std::pair<unsigned, unsigned>
AddOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index,
                                                     unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AddOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

}  // namespace detail
AddOpAdaptor::AddOpAdaptor(AddOp op)
    : AddOpAdaptor(op->getOperands(), op->getAttrDictionary(),
                   op->getRegions()) {}

::mlir::LogicalResult AddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AddOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AddOp::getLhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(
      *getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> AddOp::getRhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(
      *getODSOperands(1).begin());
}

::mlir::MutableOperandRange AddOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange =
      ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AddOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange =
      ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AddOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

void AddOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState, ::mlir::Type resultType0,
                  ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void AddOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState,
                  ::mlir::TypeRange resultTypes, ::mlir::Value lhs,
                  ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                  ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                  ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AddOp::verifyInvariantsImpl() {
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

//===----------------------------------------------------------------------===//
// ::domino::script::ConstantOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ConstantOpGenericAdaptorBase::ConstantOpGenericAdaptorBase(
    ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)
    : odsAttrs(attrs), odsRegions(regions) {
  if (odsAttrs) odsOpName.emplace("script.constant", odsAttrs.getContext());
}

std::pair<unsigned, unsigned>
ConstantOpGenericAdaptorBase::getODSOperandIndexAndLength(
    unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ConstantOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseElementsAttr ConstantOpGenericAdaptorBase::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(
                  odsAttrs.begin() + 0, odsAttrs.end() - 0,
                  ConstantOp::getValueAttrName(*odsOpName))
                  .cast<::mlir::DenseElementsAttr>();
  return attr;
}

::mlir::DenseElementsAttr ConstantOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}

}  // namespace detail
ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp op)
    : ConstantOpAdaptor(op->getOperands(), op->getAttrDictionary(),
                        op->getRegions()) {}

::mlir::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc,
                       "'script.constant' op "
                       "requires attribute 'value'");
    if (namedAttrIt->getName() == ConstantOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !((tblgen_value.isa<::mlir::DenseFPElementsAttr>() &&
                         tblgen_value.cast<::mlir::DenseElementsAttr>()
                             .getType()
                             .getElementType()
                             .isF64())))
    return emitError(loc,
                     "'script.constant' op "
                     "attribute 'value' failed to satisfy constraint: 64-bit "
                     "float elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConstantOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstantOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::DenseElementsAttr ConstantOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0,
                                              (*this)->getAttrs().end() - 0,
                                              getValueAttrName())
      .cast<::mlir::DenseElementsAttr>();
}

::mlir::DenseElementsAttr ConstantOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void ConstantOp::setValueAttr(::mlir::DenseElementsAttr attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder,
                       ::mlir::OperationState &odsState,
                       DenseElementsAttr value) {
  build(odsBuilder, odsState, value.getType(), value);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder,
                       ::mlir::OperationState &odsState,
                       ::mlir::Type resultType0,
                       ::mlir::DenseElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(resultType0);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder,
                       ::mlir::OperationState &odsState,
                       ::mlir::TypeRange resultTypes,
                       ::mlir::DenseElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                       ::mlir::TypeRange resultTypes,
                       ::mlir::ValueRange operands,
                       ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstantOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'value'");
    if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(
          __mlir_ods_local_attr_constraint_Ops0(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstantOp::verifyInvariants() {
  if (::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void ConstantOp::getEffects(
    ::llvm::SmallVectorImpl<
        ::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>
        &effects) {}

//===----------------------------------------------------------------------===//
// ::domino::script::FuncOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FuncOpGenericAdaptorBase::FuncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs,
                                                   ::mlir::RegionRange regions)
    : odsAttrs(attrs), odsRegions(regions) {
  if (odsAttrs) odsOpName.emplace("script.func", odsAttrs.getContext());
}

std::pair<unsigned, unsigned>
FuncOpGenericAdaptorBase::getODSOperandIndexAndLength(
    unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FuncOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr FuncOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(
                  odsAttrs.begin() + 1, odsAttrs.end() - 0,
                  FuncOp::getSymNameAttrName(*odsOpName))
                  .cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef FuncOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOpGenericAdaptorBase::getFunctionTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(
                  odsAttrs.begin() + 0, odsAttrs.end() - 1,
                  FuncOp::getFunctionTypeAttrName(*odsOpName))
                  .cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::FunctionType FuncOpGenericAdaptorBase::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

::mlir::ArrayAttr FuncOpGenericAdaptorBase::getArgAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(
                  odsAttrs.begin() + 0, odsAttrs.end() - 2,
                  FuncOp::getArgAttrsAttrName(*odsOpName))
                  .dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional<::mlir::ArrayAttr> FuncOpGenericAdaptorBase::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional<::mlir::ArrayAttr>(attr) : (::std::nullopt);
}

::mlir::ArrayAttr FuncOpGenericAdaptorBase::getResAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(
                  odsAttrs.begin() + 1, odsAttrs.end() - 1,
                  FuncOp::getResAttrsAttrName(*odsOpName))
                  .dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional<::mlir::ArrayAttr> FuncOpGenericAdaptorBase::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional<::mlir::ArrayAttr>(attr) : (::std::nullopt);
}

::mlir::Region &FuncOpGenericAdaptorBase::getBody() { return *odsRegions[0]; }

::mlir::RegionRange FuncOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

}  // namespace detail
FuncOpAdaptor::FuncOpAdaptor(FuncOp op)
    : FuncOpAdaptor(op->getOperands(), op->getAttrDictionary(),
                    op->getRegions()) {}

::mlir::LogicalResult FuncOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  ::mlir::Attribute tblgen_arg_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc,
                       "'script.func' op "
                       "requires attribute 'function_type'");
    if (namedAttrIt->getName() == FuncOp::getFunctionTypeAttrName(*odsOpName)) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    } else if (namedAttrIt->getName() ==
               FuncOp::getArgAttrsAttrName(*odsOpName)) {
      tblgen_arg_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_res_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc,
                       "'script.func' op "
                       "requires attribute 'sym_name'");
    if (namedAttrIt->getName() == FuncOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    } else if (namedAttrIt->getName() ==
               FuncOp::getResAttrsAttrName(*odsOpName)) {
      tblgen_res_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(
        loc,
        "'script.func' op "
        "attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_function_type &&
      !(((tblgen_function_type.isa<::mlir::TypeAttr>())) &&
        ((tblgen_function_type.cast<::mlir::TypeAttr>()
              .getValue()
              .isa<::mlir::FunctionType>())) &&
        ((tblgen_function_type.cast<::mlir::TypeAttr>()
              .getValue()
              .isa<::mlir::FunctionType>()))))
    return emitError(loc,
                     "'script.func' op "
                     "attribute 'function_type' failed to satisfy constraint: "
                     "type attribute of function type");

  if (tblgen_arg_attrs &&
      !(((tblgen_arg_attrs.isa<::mlir::ArrayAttr>())) &&
        (::llvm::all_of(tblgen_arg_attrs.cast<::mlir::ArrayAttr>(),
                        [&](::mlir::Attribute attr) {
                          return attr && ((attr.isa<::mlir::DictionaryAttr>()));
                        }))))
    return emitError(loc,
                     "'script.func' op "
                     "attribute 'arg_attrs' failed to satisfy constraint: "
                     "Array of dictionary attributes");

  if (tblgen_res_attrs &&
      !(((tblgen_res_attrs.isa<::mlir::ArrayAttr>())) &&
        (::llvm::all_of(tblgen_res_attrs.cast<::mlir::ArrayAttr>(),
                        [&](::mlir::Attribute attr) {
                          return attr && ((attr.isa<::mlir::DictionaryAttr>()));
                        }))))
    return emitError(loc,
                     "'script.func' op "
                     "attribute 'res_attrs' failed to satisfy constraint: "
                     "Array of dictionary attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FuncOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> FuncOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::Region &FuncOp::getBody() { return (*this)->getRegion(0); }

::mlir::StringAttr FuncOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1,
                                              (*this)->getAttrs().end() - 0,
                                              getSymNameAttrName())
      .cast<::mlir::StringAttr>();
}

::llvm::StringRef FuncOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOp::getFunctionTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0,
                                              (*this)->getAttrs().end() - 1,
                                              getFunctionTypeAttrName())
      .cast<::mlir::TypeAttr>();
}

::mlir::FunctionType FuncOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

::mlir::ArrayAttr FuncOp::getArgAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0,
                                              (*this)->getAttrs().end() - 2,
                                              getArgAttrsAttrName())
      .dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional<::mlir::ArrayAttr> FuncOp::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional<::mlir::ArrayAttr>(attr) : (::std::nullopt);
}

::mlir::ArrayAttr FuncOp::getResAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1,
                                              (*this)->getAttrs().end() - 1,
                                              getResAttrsAttrName())
      .dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional<::mlir::ArrayAttr> FuncOp::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional<::mlir::ArrayAttr>(attr) : (::std::nullopt);
}

void FuncOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void FuncOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(
      getSymNameAttrName(),
      ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void FuncOp::setFunctionTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getFunctionTypeAttrName(), attr);
}

void FuncOp::setFunctionType(::mlir::FunctionType attrValue) {
  (*this)->setAttr(getFunctionTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void FuncOp::setArgAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getArgAttrsAttrName(), attr);
}

void FuncOp::setResAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getResAttrsAttrName(), attr);
}

::mlir::Attribute FuncOp::removeArgAttrsAttr() {
  return (*this)->removeAttr(getArgAttrsAttrName());
}

::mlir::Attribute FuncOp::removeResAttrsAttr() {
  return (*this)->removeAttr(getResAttrsAttrName());
}

::mlir::LogicalResult FuncOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  ::mlir::Attribute tblgen_arg_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'function_type'");
    if (namedAttrIt->getName() == getFunctionTypeAttrName()) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    } else if (namedAttrIt->getName() == getArgAttrsAttrName()) {
      tblgen_arg_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_res_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    } else if (namedAttrIt->getName() == getResAttrsAttrName()) {
      tblgen_res_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops1(
          *this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops2(
          *this, tblgen_function_type, "function_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops3(
          *this, tblgen_arg_attrs, "arg_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops3(
          *this, tblgen_res_attrs, "res_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0;
    (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_Ops0(
              *this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult FuncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

//===----------------------------------------------------------------------===//
// ::domino::script::GenericCallOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GenericCallOpGenericAdaptorBase::GenericCallOpGenericAdaptorBase(
    ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)
    : odsAttrs(attrs), odsRegions(regions) {
  if (odsAttrs) odsOpName.emplace("script.generic_call", odsAttrs.getContext());
}

std::pair<unsigned, unsigned>
GenericCallOpGenericAdaptorBase::getODSOperandIndexAndLength(
    unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value
  // count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static
  // variadic operand, we need to offset by (variadicSize - 1) to get where the
  // dynamic value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr GenericCallOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr GenericCallOpGenericAdaptorBase::getCalleeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(
                  odsAttrs.begin() + 0, odsAttrs.end() - 0,
                  GenericCallOp::getCalleeAttrName(*odsOpName))
                  .cast<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::llvm::StringRef GenericCallOpGenericAdaptorBase::getCallee() {
  auto attr = getCalleeAttr();
  return attr.getValue();
}

}  // namespace detail
GenericCallOpAdaptor::GenericCallOpAdaptor(GenericCallOp op)
    : GenericCallOpAdaptor(op->getOperands(), op->getAttrDictionary(),
                           op->getRegions()) {}

::mlir::LogicalResult GenericCallOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_callee;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc,
                       "'script.generic_call' op "
                       "requires attribute 'callee'");
    if (namedAttrIt->getName() ==
        GenericCallOp::getCalleeAttrName(*odsOpName)) {
      tblgen_callee = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_callee && !((tblgen_callee.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc,
                     "'script.generic_call' op "
                     "attribute 'callee' failed to satisfy constraint: flat "
                     "symbol reference attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GenericCallOp::getODSOperandIndexAndLength(
    unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value
  // count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static
  // variadic operand, we need to offset by (variadicSize - 1) to get where the
  // dynamic value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range GenericCallOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range GenericCallOp::getInputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange GenericCallOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange =
      ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GenericCallOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GenericCallOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::FlatSymbolRefAttr GenericCallOp::getCalleeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0,
                                              (*this)->getAttrs().end() - 0,
                                              getCalleeAttrName())
      .cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef GenericCallOp::getCallee() {
  auto attr = getCalleeAttr();
  return attr.getValue();
}

void GenericCallOp::setCalleeAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getCalleeAttrName(), attr);
}

void GenericCallOp::setCallee(::llvm::StringRef attrValue) {
  (*this)->setAttr(
      getCalleeAttrName(),
      ::mlir::SymbolRefAttr::get(
          ::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void GenericCallOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState,
                          ::mlir::Type resultType0,
                          ::mlir::FlatSymbolRefAttr callee,
                          ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute(getCalleeAttrName(odsState.name), callee);
  odsState.addTypes(resultType0);
}

void GenericCallOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState,
                          ::mlir::TypeRange resultTypes,
                          ::mlir::FlatSymbolRefAttr callee,
                          ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute(getCalleeAttrName(odsState.name), callee);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GenericCallOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState,
                          ::mlir::Type resultType0, ::llvm::StringRef callee,
                          ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute(
      getCalleeAttrName(odsState.name),
      ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), callee));
  odsState.addTypes(resultType0);
}

void GenericCallOp::build(::mlir::OpBuilder &odsBuilder,
                          ::mlir::OperationState &odsState,
                          ::mlir::TypeRange resultTypes,
                          ::llvm::StringRef callee, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.addAttribute(
      getCalleeAttrName(odsState.name),
      ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), callee));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GenericCallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                          ::mlir::TypeRange resultTypes,
                          ::mlir::ValueRange operands,
                          ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GenericCallOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_callee;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'callee'");
    if (namedAttrIt->getName() == getCalleeAttrName()) {
      tblgen_callee = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops4(*this, tblgen_callee,
                                                           "callee")))
    return ::mlir::failure();
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GenericCallOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GenericCallOp::parse(::mlir::OpAsmParser &parser,
                                         ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr calleeAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> inputsTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(
          calleeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "callee",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseLParen()) return ::mlir::failure();

  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands)) return ::mlir::failure();
  if (parser.parseRParen()) return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes)) return ::mlir::failure();
  if (parser.parseColon()) return ::mlir::failure();

  ::mlir::FunctionType inputs__allResult_functionType;
  if (parser.parseType(inputs__allResult_functionType))
    return ::mlir::failure();
  inputsTypes = inputs__allResult_functionType.getInputs();
  allResultTypes = inputs__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GenericCallOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCalleeAttr());
  _odsPrinter << "(";
  _odsPrinter << getInputs();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("callee");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getInputs().getTypes(),
                                  getOperation()->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ::domino::script::MulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MulOpGenericAdaptorBase::MulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs,
                                                 ::mlir::RegionRange regions)
    : odsAttrs(attrs), odsRegions(regions) {
  if (odsAttrs) odsOpName.emplace("script.mul", odsAttrs.getContext());
}

std::pair<unsigned, unsigned>
MulOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index,
                                                     unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr MulOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

}  // namespace detail
MulOpAdaptor::MulOpAdaptor(MulOp op)
    : MulOpAdaptor(op->getOperands(), op->getAttrDictionary(),
                   op->getRegions()) {}

::mlir::LogicalResult MulOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MulOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> MulOp::getLhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(
      *getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> MulOp::getRhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(
      *getODSOperands(1).begin());
}

::mlir::MutableOperandRange MulOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange =
      ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MulOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange =
      ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MulOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

void MulOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState, ::mlir::Type resultType0,
                  ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState,
                  ::mlir::TypeRange resultTypes, ::mlir::Value lhs,
                  ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                  ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                  ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MulOp::verifyInvariantsImpl() {
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

//===----------------------------------------------------------------------===//
// ::domino::script::PrintOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PrintOpGenericAdaptorBase::PrintOpGenericAdaptorBase(
    ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)
    : odsAttrs(attrs), odsRegions(regions) {
  if (odsAttrs) odsOpName.emplace("script.print", odsAttrs.getContext());
}

std::pair<unsigned, unsigned>
PrintOpGenericAdaptorBase::getODSOperandIndexAndLength(
    unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr PrintOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

}  // namespace detail
PrintOpAdaptor::PrintOpAdaptor(PrintOp op)
    : PrintOpAdaptor(op->getOperands(), op->getAttrDictionary(),
                     op->getRegions()) {}

::mlir::LogicalResult PrintOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PrintOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PrintOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> PrintOp::getInput() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(
      *getODSOperands(0).begin());
}

::mlir::MutableOperandRange PrintOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange =
      ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PrintOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PrintOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Value input) {
  odsState.addOperands(input);
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PrintOp::verifyInvariantsImpl() {
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PrintOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PrintOp::parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(
      inputRawOperands);
  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0])) return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes)) return ::mlir::failure();
  if (parser.parseColon()) return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type)) return ::mlir::failure();
    inputRawTypes[0] = type;
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PrintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
    else
      _odsPrinter << type;
  }
}

//===----------------------------------------------------------------------===//
// ::domino::script::ReshapeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReshapeOpGenericAdaptorBase::ReshapeOpGenericAdaptorBase(
    ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)
    : odsAttrs(attrs), odsRegions(regions) {
  if (odsAttrs) odsOpName.emplace("script.reshape", odsAttrs.getContext());
}

std::pair<unsigned, unsigned>
ReshapeOpGenericAdaptorBase::getODSOperandIndexAndLength(
    unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReshapeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

}  // namespace detail
ReshapeOpAdaptor::ReshapeOpAdaptor(ReshapeOp op)
    : ReshapeOpAdaptor(op->getOperands(), op->getAttrDictionary(),
                       op->getRegions()) {}

::mlir::LogicalResult ReshapeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReshapeOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReshapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ReshapeOp::getInput() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(
      *getODSOperands(0).begin());
}

::mlir::MutableOperandRange ReshapeOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange =
      ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReshapeOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReshapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState,
                      ::mlir::Type resultType0, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(resultType0);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState,
                      ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReshapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                      ::mlir::TypeRange resultTypes,
                      ::mlir::ValueRange operands,
                      ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReshapeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops1(
              *this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReshapeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ReshapeOp::parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(
      inputRawOperands);
  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;
  if (parser.parseLParen()) return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0])) return ::mlir::failure();
  if (parser.parseColon()) return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type)) return ::mlir::failure();
    inputRawTypes[0] = type;
  }
  if (parser.parseRParen()) return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes)) return ::mlir::failure();
  if (parser.parseKeyword("to")) return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes)) return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReshapeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getInput();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
    else
      _odsPrinter << type;
  }
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

//===----------------------------------------------------------------------===//
// ::domino::script::ReturnOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReturnOpGenericAdaptorBase::ReturnOpGenericAdaptorBase(
    ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)
    : odsAttrs(attrs), odsRegions(regions) {
  if (odsAttrs) odsOpName.emplace("script.return", odsAttrs.getContext());
}

std::pair<unsigned, unsigned>
ReturnOpGenericAdaptorBase::getODSOperandIndexAndLength(
    unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value
  // count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static
  // variadic operand, we need to offset by (variadicSize - 1) to get where the
  // dynamic value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr ReturnOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

}  // namespace detail
ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp op)
    : ReturnOpAdaptor(op->getOperands(), op->getAttrDictionary(),
                      op->getRegions()) {}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(
    unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value
  // count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static
  // variadic operand, we need to offset by (variadicSize - 1) to get where the
  // dynamic value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReturnOp::getInput() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange =
      ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState) {
  build(odsBuilder, odsState, std::nullopt);
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder,
                     ::mlir::OperationState &odsState,
                     ::mlir::ValueRange input) {
  odsState.addOperands(input);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                     ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                     ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReturnOp::verifyInvariantsImpl() {
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReturnOp::verifyInvariants() {
  if (::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReturnOp::parse(::mlir::OpAsmParser &parser,
                                    ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputOperands;
  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputTypes;

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputOperands)) return ::mlir::failure();
  if (!inputOperands.empty()) {
    if (parser.parseColon()) return ::mlir::failure();

    if (parser.parseTypeList(inputTypes)) return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) return ::mlir::failure();
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReturnOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getInput().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << getInput();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getInput().getTypes();
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ReturnOp::getEffects(
    ::llvm::SmallVectorImpl<
        ::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>
        &effects) {}

//===----------------------------------------------------------------------===//
// ::domino::script::TransposeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TransposeOpGenericAdaptorBase::TransposeOpGenericAdaptorBase(
    ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions)
    : odsAttrs(attrs), odsRegions(regions) {
  if (odsAttrs) odsOpName.emplace("script.transpose", odsAttrs.getContext());
}

std::pair<unsigned, unsigned>
TransposeOpGenericAdaptorBase::getODSOperandIndexAndLength(
    unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TransposeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

}  // namespace detail
TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp op)
    : TransposeOpAdaptor(op->getOperands(), op->getAttrDictionary(),
                         op->getRegions()) {}

::mlir::LogicalResult TransposeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TransposeOp::getODSOperandIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TransposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
          std::next(getOperation()->operand_begin(),
                    valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> TransposeOp::getInput() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(
      *getODSOperands(0).begin());
}

::mlir::MutableOperandRange TransposeOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange =
      ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TransposeOp::getODSResultIndexAndLength(
    unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
          std::next(getOperation()->result_begin(),
                    valueRange.first + valueRange.second)};
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState,
                        ::mlir::Type resultType0, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(resultType0);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder,
                        ::mlir::OperationState &odsState,
                        ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                        ::mlir::TypeRange resultTypes,
                        ::mlir::ValueRange operands,
                        ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TransposeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0;
    (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(
              *this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TransposeOp::verifyInvariants() {
  if (::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TransposeOp::parse(::mlir::OpAsmParser &parser,
                                       ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(
      inputRawOperands);
  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;
  if (parser.parseLParen()) return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0])) return ::mlir::failure();
  if (parser.parseColon()) return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type)) return ::mlir::failure();
    inputRawTypes[0] = type;
  }
  if (parser.parseRParen()) return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes)) return ::mlir::failure();
  if (parser.parseKeyword("to")) return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes)) return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TransposeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getInput();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
    else
      _odsPrinter << type;
  }
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

}  // namespace script
}  // namespace domino

MLIR_DEFINE_EXPLICIT_TYPE_ID(::domino::script::AddOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::domino::script::ConstantOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::domino::script::FuncOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::domino::script::GenericCallOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::domino::script::MulOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::domino::script::PrintOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::domino::script::ReshapeOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::domino::script::ReturnOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::domino::script::TransposeOp)