#ifndef DOMINO_UTIL_MACROS_H_
#define DOMINO_UTIL_MACROS_H_

#include <cassert>

namespace domino {

#if defined(__GUNC__) || defined(__ICL) || defined(__clang__)
#define DOMINO_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define DOMINO_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define DOMINO_LIKELY(expr) (expr)
#define DOMINO_UNLIKELY(expr) (expr)
#endif

#if __has_builtin(__builtin_assume_aligned) || defined(__GNUC__)
#define DOMINO_ASSUME_ALIGNED(p, a) __builtin_assume_aligned(p, a)
#elif defined(DOMINO_BUILTIN_UNREACHABLE)
#define DOMINO_ASSUME_ALIGNED(p, a) \
  (((uintptr_t(p) % (a)) == 0) ? (p) : (DOMINO_BUILTIN_UNREACHABLE, (p)))
#else
#define DOMINO_ASSUME_ALIGNED(p, a) (p)
#endif

}  // namespace domino

#endif  // DOMINO_UTIL_MACROS_H_
