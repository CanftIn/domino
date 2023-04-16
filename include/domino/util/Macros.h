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

}

#endif  // DOMINO_UTIL_MACROS_H_
