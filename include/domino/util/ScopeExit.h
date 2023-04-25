#ifndef DOMINO_UTIL_SCOPEEXIT_H_
#define DOMINO_UTIL_SCOPEEXIT_H_

#include <type_traits>
#include <utility>

namespace domino {
namespace detail {

template <typename Callable>
class scope_exit {
  Callable ExitFunction;
  bool Engaged = true;  // False once moved-from or release()d.

 public:
  template <typename Fp>
  explicit scope_exit(Fp &&F) : ExitFunction(std::forward<Fp>(F)) {}

  scope_exit(scope_exit &&Rhs)
      : ExitFunction(std::move(Rhs.ExitFunction)), Engaged(Rhs.Engaged) {
    Rhs.release();
  }
  scope_exit(const scope_exit &) = delete;
  scope_exit &operator=(scope_exit &&) = delete;
  scope_exit &operator=(const scope_exit &) = delete;

  void release() { Engaged = false; }

  ~scope_exit() {
    if (Engaged) ExitFunction();
  }
};

}  // namespace detail

template <typename Callable>
[[nodiscard]] detail::scope_exit<std::decay_t<Callable>> make_scope_exit(
    Callable &&F) {
  return detail::scope_exit<std::decay_t<Callable>>(std::forward<Callable>(F));
}

}  // namespace domino

#endif  // DOMINO_UTIL_SCOPEEXIT_H_
