#ifndef DOMINO_SUPPORT_FILESYSTEM_UNIX_H_
#define DOMINO_SUPPORT_FILESYSTEM_UNIX_H_

#include <domino/util/Logging.h>
#include <domino/support/Chrono.h>
#include <domino/support/filesystem/Errno.h>
#include <domino/util/Twine.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <unistd.h>
#include <time.h>
#include <dlfcn.h>
#include <fcntl.h>

/// This function builds an error message into \p ErrMsg using the \p prefix
/// string and the Unix error number given by \p errnum. If errnum is -1, the
/// default then the value of errno is used.
/// Make an error message
///
/// If the error number can be converted to a string, it will be
/// separated from prefix by ": ".
static inline bool MakeErrMsg(std::string* ErrMsg, const std::string& prefix,
                              int errnum = -1) {
  if (!ErrMsg) return true;
  if (errnum == -1) errnum = errno;
  *ErrMsg = prefix + ": " + domino::sys::StrError(errnum);
  return true;
}

// Include StrError(errnum) in a fatal error message.
[[noreturn]] static inline void ReportErrnumFatal(const char* Msg, int errnum) {
  std::string ErrMsg;
  MakeErrMsg(&ErrMsg, Msg, errnum);
  // TODO: ERROR abort
}

namespace domino {
namespace sys {

/// Convert a struct timeval to a duration. Note that timeval can be used both
/// as a time point and a duration. Be sure to check what the input represents.
inline std::chrono::microseconds toDuration(const struct timeval& TV) {
  return std::chrono::seconds(TV.tv_sec) +
         std::chrono::microseconds(TV.tv_usec);
}

/// Convert a time point to struct timespec.
inline struct timespec toTimeSpec(TimePoint<> TP) {
  using namespace std::chrono;

  struct timespec RetVal;
  RetVal.tv_sec = toTimeT(TP);
  RetVal.tv_nsec = (TP.time_since_epoch() % seconds(1)).count();
  return RetVal;
}

/// Convert a time point to struct timeval.
inline struct timeval toTimeVal(TimePoint<std::chrono::microseconds> TP) {
  using namespace std::chrono;

  struct timeval RetVal;
  RetVal.tv_sec = toTimeT(TP);
  RetVal.tv_usec = (TP.time_since_epoch() % seconds(1)).count();
  return RetVal;
}

}  // namespace sys
}  // namespace domino

#endif  // DOMINO_SUPPORT_FILESYSTEM_UNIX_H_
