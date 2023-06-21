#ifndef DOMINO_UTIL_IF_CONSTEXPR_H_
#define DOMINO_UTIL_IF_CONSTEXPR_H_

#include <tuple>
#include <utility>

namespace domino {

template <bool Condition, typename TrueFunc, typename FalseFunc,
          typename... Args>
auto IfConstexprElse(TrueFunc &&true_func, FalseFunc &&false_func,
                     Args &&...args) {
  return std::get<Condition>(std::forward_as_tuple(
      std::forward<FalseFunc>(false_func), std::forward<TrueFunc>(true_func)))(
      std::forward<Args>(args)...);
}

template <bool Condition, typename Func, typename... Args>
void IfConstexpr(Func &&func, Args &&...args) {
  IfConstexprElse<Condition>(
      std::forward<Func>(func), [](auto &&...) {}, std::forward<Args>(args)...);
}

} // namespace domino

#endif // DOMINO_UTIL_IF_CONSTEXPR_H_