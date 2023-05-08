#ifndef DOMINO_SCRIPT_DIALECT_H_
#define DOMINO_SCRIPT_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace domino {
namespace script {

class ScriptDialect : public ::mlir::Dialect {
  explicit ScriptDialect(::mlir::MLIRContext *context);

  void initialize();

  friend class ::mlir::MLIRContext;

 public:
  ~ScriptDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("domino_script");
  }
};

class AddOp;

class ConstantOp;

class FuncOp;

class GenericCallOp;

class MulOp;

class PrintOp;

class ReshapeOp;

class ReturnOp;

class TransposeOp;

namespace detail {

class AddOpGenericAdaptorBase {
 protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;

 public:
  AddOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr,
                          ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(
      unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};

}  // namespace detail

template <typename RangeT>
class AddOpGenericAdaptor : public detail::AddOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AddOpGenericAdaptorBase;

 public:
  AddOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr,
                      ::mlir::RegionRange regions = {})
      : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() { return (*getODSOperands(0).begin()); }

  ValueT getRhs() { return (*getODSOperands(1).begin()); }

  RangeT getOperands() { return odsOperands; }

 private:
  RangeT odsOperands;
};

class AddOpAdaptor : public AddOpGenericAdaptor<::mlir::ValueRange> {
 public:
  using AddOpGenericAdaptor::AddOpGenericAdaptor;
  AddOpAdaptor(AddOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};

class AddOp
    : public ::mlir::Op<
          AddOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult,
          ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl,
          ::mlir::OpTrait::OpInvariants> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = AddOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AddOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("script.add");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getLhs();
  ::mlir::TypedValue<::mlir::TensorType> getRhs();
  ::mlir::MutableOperandRange getLhsMutable();
  ::mlir::MutableOperandRange getRhsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Value lhs,
                    ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type resultType0,
                    ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::Value lhs,
                    ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
};

namespace detail {

class ConstantOpGenericAdaptorBase {
 protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;

 public:
  ConstantOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr,
                               ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(
      unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseElementsAttr getValueAttr();
  ::mlir::DenseElementsAttr getValue();
};

}  // namespace detail

template <typename RangeT>
class ConstantOpGenericAdaptor : public detail::ConstantOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConstantOpGenericAdaptorBase;

 public:
  ConstantOpGenericAdaptor(RangeT values,
                           ::mlir::DictionaryAttr attrs = nullptr,
                           ::mlir::RegionRange regions = {})
      : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() { return odsOperands; }

 private:
  RangeT odsOperands;
};

class ConstantOpAdaptor : public ConstantOpGenericAdaptor<::mlir::ValueRange> {
 public:
  using ConstantOpGenericAdaptor::ConstantOpGenericAdaptor;
  ConstantOpAdaptor(ConstantOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};

class ConstantOp
    : public ::mlir::Op<
          ConstantOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult,
          ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands,
          ::mlir::OpTrait::OpInvariants,
          ::mlir::ConditionallySpeculatable::Trait,
          ::mlir::OpTrait::AlwaysSpeculatableImplTrait,
          ::mlir::MemoryEffectOpInterface::Trait> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConstantOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConstantOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("value")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getValueAttrName() { return getAttributeNameForIndex(0); }

  static ::mlir::StringAttr getValueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("script.constant");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseElementsAttr getValueAttr();
  ::mlir::DenseElementsAttr getValue();
  void setValueAttr(::mlir::DenseElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, double value);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type resultType0,
                    ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes,
                    ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<
                      ::mlir::MemoryEffects::Effect>> &effects);

 private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name,
                                                     unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() &&
           "invalid operation name");
    return name.getAttributeNames()[index];
  }
};

namespace detail {

class FuncOpGenericAdaptorBase {
 protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;

 public:
  FuncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr,
                           ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(
      unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::TypeAttr getFunctionTypeAttr();
  ::mlir::FunctionType getFunctionType();
  ::mlir::ArrayAttr getArgAttrsAttr();
  ::std::optional<::mlir::ArrayAttr> getArgAttrs();
  ::mlir::ArrayAttr getResAttrsAttr();
  ::std::optional<::mlir::ArrayAttr> getResAttrs();
  ::mlir::Region &getBody();
  ::mlir::RegionRange getRegions();
};

}  // namespace detail

template <typename RangeT>
class FuncOpGenericAdaptor : public detail::FuncOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::FuncOpGenericAdaptorBase;

 public:
  FuncOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr,
                       ::mlir::RegionRange regions = {})
      : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() { return odsOperands; }

 private:
  RangeT odsOperands;
};

class FuncOpAdaptor : public FuncOpGenericAdaptor<::mlir::ValueRange> {
 public:
  using FuncOpGenericAdaptor::FuncOpGenericAdaptor;
  FuncOpAdaptor(FuncOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};

class FuncOp
    : public ::mlir::Op<
          FuncOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands,
          ::mlir::OpTrait::OpInvariants, ::mlir::SymbolOpInterface::Trait,
          ::mlir::FunctionOpInterface::Trait,
          ::mlir::OpTrait::IsIsolatedFromAbove> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = FuncOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = FuncOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {
        ::llvm::StringRef("arg_attrs"), ::llvm::StringRef("function_type"),
        ::llvm::StringRef("res_attrs"), ::llvm::StringRef("sym_name")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getArgAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getArgAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getFunctionTypeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getFunctionTypeAttrName(
      ::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getResAttrsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getResAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getSymNameAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getSymNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("script.func");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getBody();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::TypeAttr getFunctionTypeAttr();
  ::mlir::FunctionType getFunctionType();
  ::mlir::ArrayAttr getArgAttrsAttr();
  ::std::optional<::mlir::ArrayAttr> getArgAttrs();
  ::mlir::ArrayAttr getResAttrsAttr();
  ::std::optional<::mlir::ArrayAttr> getResAttrs();
  void setSymNameAttr(::mlir::StringAttr attr);
  void setSymName(::llvm::StringRef attrValue);
  void setFunctionTypeAttr(::mlir::TypeAttr attr);
  void setFunctionType(::mlir::FunctionType attrValue);
  void setArgAttrsAttr(::mlir::ArrayAttr attr);
  void setResAttrsAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeArgAttrsAttr();
  ::mlir::Attribute removeResAttrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::llvm::StringRef name,
                    ::mlir::FunctionType type,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attrs = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();

 private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name,
                                                     unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() &&
           "invalid operation name");
    return name.getAttributeNames()[index];
  }

 public:
  /// Returns the argument types of this function.
  llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
    return getFunctionType().getInputs();
  }

  /// Returns the result types of this function.
  llvm::ArrayRef<::mlir::Type> getResultTypes() {
    return getFunctionType().getResults();
  }
};

namespace detail {

class GenericCallOpGenericAdaptorBase {
 protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;

 public:
  GenericCallOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr,
                                  ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(
      unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr getCalleeAttr();
  ::llvm::StringRef getCallee();
};

}  // namespace detail

template <typename RangeT>
class GenericCallOpGenericAdaptor
    : public detail::GenericCallOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GenericCallOpGenericAdaptorBase;

 public:
  GenericCallOpGenericAdaptor(RangeT values,
                              ::mlir::DictionaryAttr attrs = nullptr,
                              ::mlir::RegionRange regions = {})
      : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInputs() { return getODSOperands(0); }

  RangeT getOperands() { return odsOperands; }

 private:
  RangeT odsOperands;
};

class GenericCallOpAdaptor
    : public GenericCallOpGenericAdaptor<::mlir::ValueRange> {
 public:
  using GenericCallOpGenericAdaptor::GenericCallOpGenericAdaptor;
  GenericCallOpAdaptor(GenericCallOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};

class GenericCallOp
    : public ::mlir::Op<
          GenericCallOp, ::mlir::OpTrait::ZeroRegions,
          ::mlir::OpTrait::OneResult,
          ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands,
          ::mlir::OpTrait::OpInvariants> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = GenericCallOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GenericCallOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("callee")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCalleeAttrName() { return getAttributeNameForIndex(0); }

  static ::mlir::StringAttr getCalleeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("script.generic_call");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInputs();
  ::mlir::MutableOperandRange getInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::FlatSymbolRefAttr getCalleeAttr();
  ::llvm::StringRef getCallee();
  void setCalleeAttr(::mlir::FlatSymbolRefAttr attr);
  void setCallee(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::llvm::StringRef callee,
                    ::llvm::ArrayRef<::mlir::Value> arguments);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type resultType0,
                    ::mlir::FlatSymbolRefAttr callee,
                    ::mlir::ValueRange inputs);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes,
                    ::mlir::FlatSymbolRefAttr callee,
                    ::mlir::ValueRange inputs);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type resultType0,
                    ::llvm::StringRef callee, ::mlir::ValueRange inputs);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::llvm::StringRef callee,
                    ::mlir::ValueRange inputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);

 private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name,
                                                     unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() &&
           "invalid operation name");
    return name.getAttributeNames()[index];
  }
};

namespace detail {

class MulOpGenericAdaptorBase {
 protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;

 public:
  MulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr,
                          ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(
      unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};

}  // namespace detail

template <typename RangeT>
class MulOpGenericAdaptor : public detail::MulOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MulOpGenericAdaptorBase;

 public:
  MulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr,
                      ::mlir::RegionRange regions = {})
      : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() { return (*getODSOperands(0).begin()); }

  ValueT getRhs() { return (*getODSOperands(1).begin()); }

  RangeT getOperands() { return odsOperands; }

 private:
  RangeT odsOperands;
};

class MulOpAdaptor : public MulOpGenericAdaptor<::mlir::ValueRange> {
 public:
  using MulOpGenericAdaptor::MulOpGenericAdaptor;
  MulOpAdaptor(MulOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};

class MulOp
    : public ::mlir::Op<
          MulOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult,
          ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl,
          ::mlir::OpTrait::OpInvariants> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = MulOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MulOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("script.mul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getLhs();
  ::mlir::TypedValue<::mlir::TensorType> getRhs();
  ::mlir::MutableOperandRange getLhsMutable();
  ::mlir::MutableOperandRange getRhsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Value lhs,
                    ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type resultType0,
                    ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::Value lhs,
                    ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
};

namespace detail {
class PrintOpGenericAdaptorBase {
 protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;

 public:
  PrintOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr,
                            ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(
      unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
}  // namespace detail
template <typename RangeT>
class PrintOpGenericAdaptor : public detail::PrintOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PrintOpGenericAdaptorBase;

 public:
  PrintOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr,
                        ::mlir::RegionRange regions = {})
      : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() { return (*getODSOperands(0).begin()); }

  RangeT getOperands() { return odsOperands; }

 private:
  RangeT odsOperands;
};

class PrintOpAdaptor : public PrintOpGenericAdaptor<::mlir::ValueRange> {
 public:
  using PrintOpGenericAdaptor::PrintOpGenericAdaptor;
  PrintOpAdaptor(PrintOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};

class PrintOp
    : public ::mlir::Op<
          PrintOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand,
          ::mlir::OpTrait::OpInvariants> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = PrintOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PrintOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("script.print");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getInput();
  ::mlir::MutableOperandRange getInputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);

 public:
};

namespace detail {
class ReshapeOpGenericAdaptorBase {
 protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;

 public:
  ReshapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr,
                              ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(
      unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};

}  // namespace detail

template <typename RangeT>
class ReshapeOpGenericAdaptor : public detail::ReshapeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReshapeOpGenericAdaptorBase;

 public:
  ReshapeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr,
                          ::mlir::RegionRange regions = {})
      : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() { return (*getODSOperands(0).begin()); }

  RangeT getOperands() { return odsOperands; }

 private:
  RangeT odsOperands;
};

class ReshapeOpAdaptor : public ReshapeOpGenericAdaptor<::mlir::ValueRange> {
 public:
  using ReshapeOpGenericAdaptor::ReshapeOpGenericAdaptor;
  ReshapeOpAdaptor(ReshapeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};

class ReshapeOp
    : public ::mlir::Op<
          ReshapeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult,
          ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand,
          ::mlir::OpTrait::OpInvariants> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReshapeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReshapeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("script.reshape");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getInput();
  ::mlir::MutableOperandRange getInputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type resultType0,
                    ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
};

namespace detail {

class ReturnOpGenericAdaptorBase {
 protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;

 public:
  ReturnOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr,
                             ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(
      unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};

}  // namespace detail

template <typename RangeT>
class ReturnOpGenericAdaptor : public detail::ReturnOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReturnOpGenericAdaptorBase;

 public:
  ReturnOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr,
                         ::mlir::RegionRange regions = {})
      : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInput() { return getODSOperands(0); }

  RangeT getOperands() { return odsOperands; }

 private:
  RangeT odsOperands;
};

class ReturnOpAdaptor : public ReturnOpGenericAdaptor<::mlir::ValueRange> {
 public:
  using ReturnOpGenericAdaptor::ReturnOpGenericAdaptor;
  ReturnOpAdaptor(ReturnOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};

class ReturnOp
    : public ::mlir::Op<
          ReturnOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands,
          ::mlir::OpTrait::HasParent<FuncOp>::Impl,
          ::mlir::OpTrait::OpInvariants,
          ::mlir::ConditionallySpeculatable::Trait,
          ::mlir::OpTrait::AlwaysSpeculatableImplTrait,
          ::mlir::MemoryEffectOpInterface::Trait,
          ::mlir::OpTrait::IsTerminator> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReturnOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReturnOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("script.return");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInput();
  ::mlir::MutableOperandRange getInputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::ValueRange input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<
                      ::mlir::MemoryEffects::Effect>> &effects);

 public:
  bool hasOperand() { return getNumOperands() != 0; }
};

namespace detail {

class TransposeOpGenericAdaptorBase {
 protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;

 public:
  TransposeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr,
                                ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(
      unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};

}  // namespace detail

template <typename RangeT>
class TransposeOpGenericAdaptor : public detail::TransposeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TransposeOpGenericAdaptorBase;

 public:
  TransposeOpGenericAdaptor(RangeT values,
                            ::mlir::DictionaryAttr attrs = nullptr,
                            ::mlir::RegionRange regions = {})
      : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {
        std::next(odsOperands.begin(), valueRange.first),
        std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() { return (*getODSOperands(0).begin()); }

  RangeT getOperands() { return odsOperands; }

 private:
  RangeT odsOperands;
};

class TransposeOpAdaptor
    : public TransposeOpGenericAdaptor<::mlir::ValueRange> {
 public:
  using TransposeOpGenericAdaptor::TransposeOpGenericAdaptor;
  TransposeOpAdaptor(TransposeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};

class TransposeOp
    : public ::mlir::Op<
          TransposeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult,
          ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl,
          ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand,
          ::mlir::OpTrait::OpInvariants> {
 public:
  using Op::Op;
  using Op::print;
  using Adaptor = TransposeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TransposeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() { return {}; }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("script.transpose");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getInput();
  ::mlir::MutableOperandRange getInputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState, ::mlir::Type resultType0,
                    ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder,
                    ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState,
                    ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands,
                    ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                   ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
};

}  // namespace script
}  // namespace domino

MLIR_DECLARE_EXPLICIT_TYPE_ID(::domino::script::ScriptDialect)

MLIR_DECLARE_EXPLICIT_TYPE_ID(::domino::script::AddOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::domino::script::ConstantOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::domino::script::FuncOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::domino::script::GenericCallOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::domino::script::MulOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::domino::script::PrintOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::domino::script::ReshapeOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::domino::script::TransposeOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::domino::script::ReturnOp)

#endif  // DOMINO_SCRIPT_DIALECT_H_