#ifndef DOMINO_UTIL_STLEXTRAS_H_
#define DOMINO_UTIL_STLEXTRAS_H_

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

}  // namespace domino

#endif  // DOMINO_UTIL_STLEXTRAS_H_