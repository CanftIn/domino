#ifndef DOMINO_SUPPORT_CASTING_H_
#define DOMINO_SUPPORT_CASTING_H_

#include <domino/support/TypeTraits.h>

#include <cassert>
#include <memory>
#include <optional>
#include <type_traits>

namespace domino {

template <typename From>
struct simplify_type {
  using SimpleType = From;

  static SimpleType& getSimplifiedValue(From& Val) { return Val; }
};

template <typename From>
struct simplify_type<const From> {
  using NonConstSimpleType = typename simplify_type<From>::SimpleType;
  using SimpleType = typename add_const_past_pointer<NonConstSimpleType>::type;
  using RetType =
      typename add_lvalue_reference_if_not_pointer<SimpleType>::type;

  static RetType getSimplifiedValue(const From& Val) {
    return simplify_type<From>::getSimplifiedValue(const_cast<From&>(Val));
  }
};

template <typename From>
struct is_simple_type {
  static const bool value =
      std::is_same<From, typename simplify_type<From>::SimpleType>::value;
};

template <typename To, typename From, typename Enable = void>
struct isa_impl {
  static inline bool doit(const From& Val) { return To::classof(&Val); }
};

template <typename To, typename From>
struct isa_impl<To, From, std::enable_if_t<std::is_base_of<To, From>::value>> {
  static inline bool doit(const From&) { return true; }
};

template <typename To, typename From>
struct isa_impl_cl {
  static inline bool doit(const From& Val) {
    return isa_impl<To, From>::doit(Val);
  }
};

template <typename To, typename From>
struct isa_impl_cl<To, const From> {
  static inline bool doit(const From& Val) {
    return isa_impl<To, From>::doit(Val);
  }
};

template <typename To, typename From>
struct isa_impl_cl<To, const std::unique_ptr<From>> {
  static inline bool doit(const std::unique_ptr<From>& Val) {
    assert(Val && "isa<> used on a null pointer");
    return isa_impl<To, From>::doit(*Val);
  }
};

template <typename To, typename From>
struct isa_impl_cl<To, From*> {
  static inline bool doit(const From* Val) {
    assert(Val && "isa<> used on a null pointer");
    return isa_impl<To, From>::doit(*Val);
  }
};

template <typename To, typename From>
struct isa_impl_cl<To, const From*> {
  static inline bool doit(const From* Val) {
    assert(Val && "isa<> used on a null pointer");
    return isa_impl<To, From>::doit(*Val);
  }
};

template <typename To, typename From>
struct isa_impl_cl<To, From* const> {
  static inline bool doit(const From* Val) {
    assert(Val && "isa<> used on a null pointer");
    return isa_impl<To, From>::doit(*Val);
  }
};

template <typename To, typename From>
struct isa_impl_cl<To, const From* const> {
  static inline bool doit(const From* Val) {
    assert(Val && "isa<> used on a null pointer");
    return isa_impl<To, From>::doit(*Val);
  }
};

template <typename To, typename From, typename SimpleFrom>
struct isa_impl_wrap {
  static inline bool doit(const From& Val) {
    return isa_impl_wrap<To, SimpleFrom,
                         typename simplify_type<SimpleFrom>::SimpleType>::
        doit(simplify_type<const From>::getSimplifiedValue(Val));
  }
};

template <typename To, typename From>
struct isa_impl_wrap<To, From, From> {
  static inline bool doit(const From& Val) {
    return isa_impl_cl<To, From>::doit(Val);
  }
};

template <typename To, typename From>
struct cast_retty;

template <typename To, typename From>
struct cast_retty_impl {
  using ret_type = To&;
};

template <typename To, typename From>
struct cast_retty_impl<To, const From> {
  using ret_type = const To&;
};

template <typename To, typename From>
struct cast_retty_impl<To, From*> {
  using ret_type = To*;
};

template <typename To, typename From>
struct cast_retty_impl<To, const From*> {
  using ret_type = const To*;
};

template <typename To, typename From>
struct cast_retty_impl<To, const From* const> {
  using ret_type = const To*;
};

template <typename To, typename From>
struct cast_retty_impl<To, std::unique_ptr<From>> {
 private:
  using PointerType = typename cast_retty_impl<To, From*>::ret_type;
  using ResultType = std::remove_pointer_t<PointerType>;

 public:
  using ret_type = std::unique_ptr<ResultType>;
};

template <typename To, typename From, typename SimpleFrom>
struct cast_retty_wrap {
  using ret_type = typename cast_retty<To, SimpleFrom>::ret_type;
};

template <typename To, typename From>
struct cast_retty_wrap<To, From, From> {
  using ret_type = typename cast_retty_impl<To, From>::ret_type;
};

template <typename To, typename From>
struct cast_retty {
  using ret_type = typename cast_retty_wrap<
      To, From, typename simplify_type<From>::SimpleType>::ret_type;
};

template <typename To, typename From, class SimpleFrom>
struct cast_convert_val {
  static typename cast_retty<To, From>::ret_type doit(const From& Val) {
    return cast_convert_val<To, SimpleFrom,
                            typename simplify_type<SimpleFrom>::SimpleType>::
        doit(simplify_type<From>::getSimplifiedValue(const_cast<From&>(Val)));
  }
};

template <typename To, typename From>
struct cast_convert_val<To, From, From> {
  static typename cast_retty<To, From>::ret_type doit(const From& Val) {
    return *(std::remove_reference_t<
             typename cast_retty<To, From>::ret_type>*)&const_cast<From&>(Val);
  }
};

template <typename To, typename From>
struct cast_convert_val<To, From*, From*> {
  static typename cast_retty<To, From*>::ret_type doit(From* Val) {
    return (typename cast_retty<To, From*>::ret_type) const_cast<From*>(Val);
  }
};

template <typename To, typename From, typename Enable = void>
struct CastIsPossible {
  static inline bool isPossible(const From& f) {
    return isa_impl_wrap<
        To, const From,
        typename simplify_type<const From>::SimpleType>::doit(f);
  }
};

template <typename To, typename From>
struct CastIsPossible<To, std::optional<From>> {
  static inline bool isPossible(const std::optional<From>& f) {
    assert(f && "CastIsPossible::isPossible called on a nullptr!");
    return isa_impl_wrap<
        To, const From,
        typename simplify_type<const From>::SimpleType>::doit(*f);
  }
};

template <typename To, typename From>
struct CastIsPossible<To, From,
                      std::enable_if_t<std::is_base_of<To, From>::value>> {
  static inline bool isPossible(const From&) { return true; }
};

template <typename To>
struct NullableValueCastFailed {
  static To castFailed() { return To(nullptr); }
};

template <typename To, typename From, typename Derived>
struct DefaultDoCastIfPossible {
  static To doCastIfPossible(From f) {
    if (!Derived::isPossible(f)) return Derived::castFailed();
    return Derived::doCast(f);
  }
};

namespace detail {

template <typename OptionalDerived, typename Default>
using SelfType = std::conditional_t<std::is_same_v<OptionalDerived, void>,
                                    Default, OptionalDerived>;

}  // namespace detail

template <typename To, typename From, typename Derived = void>
struct ValueFromPointerCast
    : public CastIsPossible<To, From*>,
      public NullableValueCastFailed<To>,
      public DefaultDoCastIfPossible<
          To, From*,
          detail::SelfType<Derived, ValueFromPointerCast<To, From>>> {
  static inline To doCast(From* f) { return To(f); }
};

template <typename To, typename From, typename Derived = void>
struct UniquePtrCast : public CastIsPossible<To, From*> {
  using Self = detail::SelfType<Derived, UniquePtrCast<To, From>>;
  using CastResultType = std::unique_ptr<
      std::remove_reference_t<typename cast_retty<To, From>::ret_type>>;

  static inline CastResultType doCast(std::unique_ptr<From>&& f) {
    return CastResultType((typename CastResultType::element_type*)f.release());
  }

  static inline CastResultType castFailed() { return CastResultType(nullptr); }

  static inline CastResultType doCastIfPossible(std::unique_ptr<From>&& f) {
    if (!Self::isPossible(f)) return castFailed();
    return doCast(f);
  }
};

template <typename To, typename From, typename Derived = void>
struct OptionalValueCast
    : public CastIsPossible<To, From>,
      public DefaultDoCastIfPossible<
          std::optional<To>, From,
          detail::SelfType<Derived, OptionalValueCast<To, From>>> {
  static inline std::optional<To> castFailed() { return std::optional<To>{}; }
  static inline std::optional<To> doCast(const From& f) { return To(f); }
};

/// template<> struct CastInfo<foo, bar> {
///   ...verbose implementation...
/// };
///
/// template<> struct CastInfo<foo, const bar> : public
///        ConstStrippingForwardingCast<foo, const bar, CastInfo<foo, bar>> {};
///
template <typename To, typename From, typename ForwardTo>
struct ConstStrippingForwardingCast {
  using DecayedFrom = std::remove_cv_t<std::remove_pointer_t<From>>;
  using NonConstFrom =
      std::conditional_t<std::is_pointer_v<From>, DecayedFrom*, DecayedFrom&>;

  static inline bool isPossible(const From& f) {
    return ForwardTo::isPossible(const_cast<NonConstFrom>(f));
  }

  static inline decltype(auto) castFailed() { return ForwardTo::castFailed(); }

  static inline decltype(auto) doCast(const From& f) {
    return ForwardTo::doCast(const_cast<NonConstFrom>(f));
  }

  static inline decltype(auto) doCastIfPossible(const From& f) {
    return ForwardTo::doCastIfPossible(const_cast<NonConstFrom>(f));
  }
};

/// template <> struct CastInfo<foo, bar *> { ... verbose implementation... };
///
/// template <>
/// struct CastInfo<foo, bar>
///     : public ForwardToPointerCast<foo, bar, CastInfo<foo, bar *>> {};
///
template <typename To, typename From, typename ForwardTo>
struct ForwardToPointerCast {
  static inline bool isPossible(const From& f) {
    return ForwardTo::isPossible(&f);
  }

  static inline decltype(auto) doCast(const From& f) {
    return *ForwardTo::doCast(&f);
  }
};

//===----------------------------------------------------------------------===//
// CastInfo
//===----------------------------------------------------------------------===//

/// In order to specialize different behaviors, specify different functions in
/// your CastInfo specialization.
/// For isa<> customization, provide:
///
///   `static bool isPossible(const From &f)`
///
/// For cast<> customization, provide:
///
///  `static To doCast(const From &f)`
///
/// For dyn_cast<> and the *_if_present<> variants' customization, provide:
///
///  `static To castFailed()` and `static To doCastIfPossible(const From &f)`
///
/// Your specialization might look something like this:
///
///  template<> struct CastInfo<foo, bar> : public CastIsPossible<foo, bar> {
///    static inline foo doCast(const bar &b) {
///      return foo(const_cast<bar &>(b));
///    }
///    static inline foo castFailed() { return foo(); }
///    static inline foo doCastIfPossible(const bar &b) {
///      if (!CastInfo<foo, bar>::isPossible(b))
///        return castFailed();
///      return doCast(b);
///    }
///  };

template <typename To, typename From, typename Enable = void>
struct CastInfo : public CastIsPossible<To, From> {
  using Self = CastInfo<To, From, Enable>;

  using CastReturnType = typename cast_retty<To, From>::ret_type;

  static inline CastReturnType doCast(const From& f) {
    return cast_convert_val<
        To, From,
        typename simplify_type<From>::SimpleType>::doit(const_cast<From&>(f));
  }

  static inline CastReturnType castFailed() { return CastReturnType(nullptr); }

  static inline CastReturnType doCastIfPossible(const From& f) {
    if (!Self::isPossible(f)) return castFailed();
    return doCast(f);
  }
};

template <typename To, typename From>
struct CastInfo<To, From, std::enable_if_t<!is_simple_type<From>::value>> {
  using Self = CastInfo<To, From>;

  using SimpleFrom = typename simplify_type<From>::SimpleType;

  using SimplifiedSelf = CastInfo<To, SimpleFrom>;

  static inline bool isPossible(From& f) {
    return SimplifiedSelf::isPossible(
        simplify_type<From>::getSimplifiedValue(f));
  }

  static inline decltype(auto) doCast(From& f) {
    return SimplifiedSelf::doCast(simplify_type<From>::getSimplifiedValue(f));
  }

  static inline decltype(auto) castFailed() {
    return SimplifiedSelf::castFailed();
  }

  static inline decltype(auto) doCastIfPossible(From& f) {
    return SimplifiedSelf::doCastIfPossible(
        simplify_type<From>::getSimplifiedValue(f));
  }
};

template <typename To, typename From>
struct CastInfo<To, std::unique_ptr<From>> : public UniquePtrCast<To, From> {};

template <typename To, typename From>
struct CastInfo<To, std::optional<From>> : public OptionalValueCast<To, From> {
};

template <typename To, typename From>
[[nodiscard]] inline bool isa(const From& Val) {
  return CastInfo<To, const From>::isPossible(Val);
}

template <typename First, typename Second, typename... Rest, typename From>
[[nodiscard]] inline bool isa(const From& Val) {
  return isa<First>(Val) || isa<Second, Rest...>(Val);
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) cast(const From& Val) {
  assert(isa<To>(Val) && "cast<Ty>() argument of incompatible type!");
  return CastInfo<To, const From>::doCast(Val);
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) cast(From& Val) {
  assert(isa<To>(Val) && "cast<Ty>() argument of incompatible type!");
  return CastInfo<To, From>::doCast(Val);
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) cast(From* Val) {
  assert(isa<To>(Val) && "cast<Ty>() argument of incompatible type!");
  return CastInfo<To, From*>::doCast(Val);
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) cast(std::unique_ptr<From>&& Val) {
  assert(isa<To>(Val) && "cast<Ty>() argument of incompatible type!");
  return CastInfo<To, std::unique_ptr<From>>::doCast(std::move(Val));
}

template <typename T>
constexpr bool IsNullable =
    std::is_pointer_v<T> || std::is_constructible_v<T, std::nullptr_t>;

template <typename T, typename Enable = void>
struct ValueIsPresent {
  using UnwrappedType = T;
  static inline bool isPresent(const T& t) { return true; }
  static inline decltype(auto) unwrapValue(T& t) { return t; }
};

template <typename T>
struct ValueIsPresent<std::optional<T>> {
  using UnwrappedType = T;
  static inline bool isPresent(const std::optional<T>& t) {
    return t.has_value();
  }
  static inline decltype(auto) unwrapValue(std::optional<T>& t) { return *t; }
};

template <typename T>
struct ValueIsPresent<T, std::enable_if_t<IsNullable<T>>> {
  using UnwrappedType = T;
  static inline bool isPresent(const T& t) { return t != T(nullptr); }
  static inline decltype(auto) unwrapValue(T& t) { return t; }
};

namespace detail {

template <typename T>
inline bool isPresent(const T& t) {
  return ValueIsPresent<typename simplify_type<T>::SimpleType>::isPresent(
      simplify_type<T>::getSimplifiedValue(const_cast<T&>(t)));
};

template <typename T>
inline decltype(auto) unwrapValue(T& t) {
  return ValueIsPresent<T>::unwrapValue(t);
}

}  // namespace detail

///  if (const Instruction *I = dyn_cast<Instruction>(myVal)) { ... }

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) dyn_cast(const From& Val) {
  assert(detail::isPresent(Val) && "dyn_cast on a non-existent value");
  return CastInfo<To, const From>::doCastIfPossible(Val);
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) dyn_cast(From& Val) {
  assert(detail::isPresent(Val) && "dyn_cast on a non-existent value");
  return CastInfo<To, From>::doCastIfPossible(Val);
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) dyn_cast(From* Val) {
  assert(detail::isPresent(Val) && "dyn_cast on a non-existent value");
  return CastInfo<To, From*>::doCastIfPossible(Val);
}

template <typename To, typename From>
[[nodiscard]] inline decltype(auto) dyn_cast(std::unique_ptr<From>&& Val) {
  assert(detail::isPresent(Val) && "dyn_cast on a non-existent value");
  return CastInfo<To, std::unique_ptr<From>>::doCastIfPossible(
      std::forward<std::unique_ptr<From>&&>(Val));
}

template <typename... X, class Y>
[[nodiscard]] inline bool isa_and_present(const Y& Val) {
  if (!detail::isPresent(Val)) return false;
  return isa<X...>(Val);
}

template <typename... X, class Y>
[[nodiscard]] inline bool isa_and_nonnull(const Y& Val) {
  return isa_and_present<X...>(Val);
}

template <typename X, typename Y>
[[nodiscard]] inline auto cast_if_present(const Y& Val) {
  if (!detail::isPresent(Val)) return CastInfo<X, const Y>::castFailed();
  assert(isa<X>(Val) && "cast_if_present<Ty>() argument of incompatible type!");
  return cast<X>(detail::unwrapValue(Val));
}

template <typename X, typename Y>
[[nodiscard]] inline auto cast_if_present(Y& Val) {
  if (!detail::isPresent(Val)) return CastInfo<X, Y>::castFailed();
  assert(isa<X>(Val) && "cast_if_present<Ty>() argument of incompatible type!");
  return cast<X>(detail::unwrapValue(Val));
}

template <typename X, typename Y>
[[nodiscard]] inline auto cast_if_present(Y* Val) {
  if (!detail::isPresent(Val)) return CastInfo<X, Y*>::castFailed();
  assert(isa<X>(Val) && "cast_if_present<Ty>() argument of incompatible type!");
  return cast<X>(detail::unwrapValue(Val));
}

template <typename X, typename Y>
[[nodiscard]] inline auto cast_if_present(std::unique_ptr<Y>&& Val) {
  if (!detail::isPresent(Val)) return UniquePtrCast<X, Y>::castFailed();
  return UniquePtrCast<X, Y>::doCast(std::move(Val));
}

template <typename X, typename Y>
auto cast_or_null(const Y& Val) {
  return cast_if_present<X>(Val);
}

template <typename X, typename Y>
auto cast_or_null(Y& Val) {
  return cast_if_present<X>(Val);
}

template <typename X, typename Y>
auto cast_or_null(Y* Val) {
  return cast_if_present<X>(Val);
}

template <typename X, typename Y>
auto cast_or_null(std::unique_ptr<Y>&& Val) {
  return cast_if_present<X>(std::move(Val));
}

template <typename X, typename Y>
auto dyn_cast_if_present(const Y& Val) {
  if (!detail::isPresent(Val)) return CastInfo<X, const Y>::castFailed();
  return CastInfo<X, const Y>::doCastIfPossible(detail::unwrapValue(Val));
}

template <typename X, typename Y>
auto dyn_cast_if_present(Y& Val) {
  if (!detail::isPresent(Val)) return CastInfo<X, Y>::castFailed();
  return CastInfo<X, Y>::doCastIfPossible(detail::unwrapValue(Val));
}

template <typename X, typename Y>
auto dyn_cast_or_null(const Y& Val) {
  return dyn_cast_if_present<X>(Val);
}

template <typename X, typename Y>
auto dyn_cast_or_null(Y& Val) {
  return dyn_cast_if_present<X>(Val);
}

template <typename X, typename Y>
auto dyn_cast_or_null(Y* Val) {
  return dyn_cast_if_present<X>(Val);
}

template <class X, class Y>
[[nodiscard]] inline typename CastInfo<X, std::unique_ptr<Y>>::CastResultType
unique_dyn_cast(std::unique_ptr<Y>& Val) {
  if (!isa<X>(Val)) return nullptr;
  return cast<X>(std::move(Val));
}

template <class X, class Y>
[[nodiscard]] inline auto unique_dyn_cast(std::unique_ptr<Y>&& Val) {
  return unique_dyn_cast<X, Y>(Val);
}

template <class X, class Y>
[[nodiscard]] inline typename CastInfo<X, std::unique_ptr<Y>>::CastResultType
unique_dyn_cast_or_null(std::unique_ptr<Y>& Val) {
  if (!Val) return nullptr;
  return unique_dyn_cast<X, Y>(Val);
}

template <class X, class Y>
[[nodiscard]] inline auto unique_dyn_cast_or_null(std::unique_ptr<Y>&& Val) {
  return unique_dyn_cast_or_null<X, Y>(Val);
}

}  // namespace domino

#endif  // DOMINO_SUPPORT_CASTING_H_