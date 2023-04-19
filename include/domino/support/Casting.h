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

template <typename To, typename FromTy>
struct isa_impl_wrap<To, FromTy, FromTy> {
  static inline bool doit(const FromTy& Val) {
    return isa_impl_cl<To, FromTy>::doit(Val);
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
        To, From, typename simplify_type<const From>::SimpleType>::doit(*f);
  }
};

template <typename To, typename From>
struct CastIsPossible<To, From,
                      std::enable_if_t<std::is_base_of<To, From>::value>> {
  static inline bool isPossible(const From&) { return true; }
};

template <typename To, typename From>
[[nodiscard]] inline bool isa(const From& Val) {
  return CastIsPossible<To, const From>::isPossible(Val);
}

}  // namespace domino

#endif  // DOMINO_SUPPORT_CASTING_H_