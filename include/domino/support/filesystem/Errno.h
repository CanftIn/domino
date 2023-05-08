#ifndef DOMINO_SUPPORT_FILESYSTEM_ERRNO_H_
#define DOMINO_SUPPORT_FILESYSTEM_ERRNO_H_

#include <cerrno>
#include <string>

namespace domino {
namespace sys {

/// Returns a string representation of the errno value, using whatever
/// thread-safe variant of strerror() is available.  Be sure to call this
/// immediately after the function that set errno, or errno may have been
/// overwritten by an intervening call.
std::string StrError();

/// Like the no-argument version above, but uses \p errnum instead of errno.
std::string StrError(int errnum);

template <typename FailT, typename Fun, typename... Args>
inline decltype(auto) RetryAfterSignal(const FailT &Fail, const Fun &F,
                                       const Args &...As) {
  decltype(F(As...)) Res;
  do {
    errno = 0;
    Res = F(As...);
  } while (Res == Fail && errno == EINTR);
  return Res;
}

}  // namespace sys
}  // namespace domino

#endif  // DOMINO_SUPPORT_FILESYSTEM_ERRNO_H_