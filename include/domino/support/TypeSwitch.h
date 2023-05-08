#ifndef DOMINO_SUPPORT_TYPESWITCH_H_
#define DOMINO_SUPPORT_TYPESWITCH_H_

#include <domino/support/Casting.h>
#include <domino/util/STLExtras.h>
#include <domino/util/Macros.h>

#include <optional>

namespace domino {

namespace detail {

template <typename DerivedT, typename T>
class TypeSwitchBase {
 public:
  TypeSwitchBase(const T &value) : value(value) {}
  TypeSwitchBase(TypeSwitchBase &&other) : value(other.value) {}
  ~TypeSwitchBase() = default;

  TypeSwitchBase(const TypeSwitchBase &) = delete;
  void operator=(const TypeSwitchBase &) = delete;
  void operator=(TypeSwitchBase &&other) = delete;

  template <typename CaseT, typename CaseT2, typename... CaseTs,
            typename CallableT>
  DOMINO_ATTRIBUTE_ALWAYS_INLINE DOMINO_ATTRIBUTE_NODEBUG
  DerivedT& Case(CallableT &&caseFn) {
    DerivedT &derived = static_cast<DerivedT &>(*this);
    return derived.template Case<CaseT>(caseFn)
        .template Case<CaseT2, CaseTs...>(caseFn);
  }

  template <typename CallableT>
  DerivedT &Case(CallableT &&caseFn) {
    using Traits = function_traits<std::decay_t<CallableT>>;
    using CaseT = std::remove_cv_t<std::remove_pointer_t<
        std::remove_reference_t<typename Traits::template arg_t<0>>>>;

    DerivedT &derived = static_cast<DerivedT &>(*this);
    return derived.template Case<CaseT>(std::forward<CallableT>(caseFn));
  }

 protected:
  /// Trait to check whether `ValueT` provides a 'dyn_cast' method with type
  /// `CastT`.
  template <typename ValueT, typename CastT>
  using has_dyn_cast_t =
      decltype(std::declval<ValueT &>().template dyn_cast<CastT>());

  /// Attempt to dyn_cast the given `value` to `CastT`. This overload is
  /// selected if `value` already has a suitable dyn_cast method.
  template <typename CastT, typename ValueT>
  static decltype(auto) castValue(
      ValueT &&value,
      std::enable_if_t<is_detected<has_dyn_cast_t, ValueT, CastT>::value> * =
          nullptr) {
    return value.template dyn_cast<CastT>();
  }

  /// Attempt to dyn_cast the given `value` to `CastT`. This overload is
  /// selected if llvm::dyn_cast should be used.
  template <typename CastT, typename ValueT>
  static decltype(auto) castValue(
      ValueT &&value,
      std::enable_if_t<!is_detected<has_dyn_cast_t, ValueT, CastT>::value> * =
          nullptr) {
    return dyn_cast<CastT>(value);
  }

  const T value;
};

}  // namespace detail

/// Example:
///  Operation *op = ...;
///  LogicalResult result = TypeSwitch<Operation *, LogicalResult>(op)
///    .Case<ConstantOp>([](ConstantOp op) { ... })
///    .Default([](Operation *op) { ... });
///
template <typename T, typename ResultT = void>
class TypeSwitch : public detail::TypeSwitchBase<TypeSwitch<T, ResultT>, T> {
 public:
  using BaseT = detail::TypeSwitchBase<TypeSwitch<T, ResultT>, T>;
  using BaseT::BaseT;
  using BaseT::Case;
  TypeSwitch(TypeSwitch &&other) = default;

  /// Add a case on the given type.
  template <typename CaseT, typename CallableT>
  TypeSwitch<T, ResultT> &Case(CallableT &&caseFn) {
    if (result) return *this;

    // Check to see if CaseT applies to 'value'.
    if (auto caseValue = BaseT::template castValue<CaseT>(this->value))
      result.emplace(caseFn(caseValue));
    return *this;
  }

  /// As a default, invoke the given callable within the root value.
  template <typename CallableT>
  [[nodiscard]] ResultT Default(CallableT &&defaultFn) {
    if (result) return std::move(*result);
    return defaultFn(this->value);
  }
  /// As a default, return the given value.
  [[nodiscard]] ResultT Default(ResultT defaultResult) {
    if (result) return std::move(*result);
    return defaultResult;
  }

  [[nodiscard]] operator ResultT() {
    assert(result && "Fell off the end of a type-switch");
    return std::move(*result);
  }

 private:
  /// The pointer to the result of this switch statement, once known,
  /// null before that.
  std::optional<ResultT> result;
};

/// Specialization of TypeSwitch for void returning callables.
template <typename T>
class TypeSwitch<T, void>
    : public detail::TypeSwitchBase<TypeSwitch<T, void>, T> {
 public:
  using BaseT = detail::TypeSwitchBase<TypeSwitch<T, void>, T>;
  using BaseT::BaseT;
  using BaseT::Case;
  TypeSwitch(TypeSwitch &&other) = default;

  /// Add a case on the given type.
  template <typename CaseT, typename CallableT>
  TypeSwitch<T, void> &Case(CallableT &&caseFn) {
    if (foundMatch) return *this;

    // Check to see if any of the types apply to 'value'.
    if (auto caseValue = BaseT::template castValue<CaseT>(this->value)) {
      caseFn(caseValue);
      foundMatch = true;
    }
    return *this;
  }

  /// As a default, invoke the given callable within the root value.
  template <typename CallableT>
  void Default(CallableT &&defaultFn) {
    if (!foundMatch) defaultFn(this->value);
  }

 private:
  /// A flag detailing if we have already found a match.
  bool foundMatch = false;
};

}  // namespace domino

#endif  // DOMINO_SUPPORT_TYPESWITCH_H_