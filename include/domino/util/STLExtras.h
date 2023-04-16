#ifndef DOMINO_UTIL_STLEXTRAS_H_
#define DOMINO_UTIL_STLEXTRAS_H_

#include <cstdint>
#include <optional>
#include <type_traits>

namespace domino {

template <typename T>
struct remove_cvref {
  using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <typename T>
using remove_cvref_t = typename domino::remove_cvref<T>::type;

template <typename T, typename Function>
auto transformOptional(const std::optional<T>& O, const Function& F)
    -> std::optional<decltype(F(*O))> {
  if (O) return F(*O);
  return std::nullopt;
}

template <typename T, typename Function>
auto transformOptional(std::optional<T>&& O, const Function& F)
    -> std::optional<decltype(F(*std::move(O)))> {
  if (O) return F(*std::move(O));
  return std::nullopt;
}

template <typename Fn>
class function_ref;

template <typename Ret, typename... Params>
class function_ref<Ret(Params...)> {
  Ret (*callback)(intptr_t callable, Params... params) = nullptr;
  intptr_t callable;

  template <typename Callable>
  static Ret callback_fn(intptr_t callable, Params... params) {
    return (*reinterpret_cast<Callable*>(callable))(
        std::forward<Params>(params)...);
  }

 public:
  function_ref() = default;
  function_ref(std::nullptr_t) {}

  template <typename Callable>
  function_ref(
    Callable&& callable,
    std::enable_if_t<!std::is_same<remove_cvref_t<Callable>,
                                   function_ref>::value>* = nullptr,
    std::enable_if_t<std::is_void<Ret>::value ||
                     std::is_convertible<decltype(std::declval<Callable>()(
                                             std::declval<Params>()...)),
                                         Ret>::value>* = nullptr)
      : callback(callback_fn<std::remove_reference_t<Callable>>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}  
  
  Ret operator()(Params... params) const {
    return callback(callable, std::forward<Params>(params)...);
  }

  explicit operator bool() const { return callback; }
};

}  // namespace domino

#endif  // DOMINO_UTIL_STLEXTRAS_H_