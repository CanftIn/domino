#include <domino/support/filesystem/Errno.h>
#include <domino/support/raw_ostream.h>

#include <cstring>

namespace domino {
namespace sys {

std::string StrError() { return StrError(errno); }

std::string StrError(int errnum) {
  std::string str;
  if (errnum == 0) return str;

  raw_string_ostream stream(str);
  stream << "Error #" << errnum;
  stream.flush();
  return str;
}

}  // namespace sys
}  // namespace domino
