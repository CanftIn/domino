#ifndef DOMINO_SUPPORT_PROGRAM_H_
#define DOMINO_SUPPORT_PROGRAM_H_

#include <domino/support/Errc.h>
#include <domino/support/FileSystem.h>

namespace domino {

const char EnvPathSeparator = ':';

class Program {};

namespace sys {

  std::error_code ChangeStdinToBinary();
  std::error_code ChangeStdoutToBinary();

  std::error_code ChangeStdinMode(fs::OpenFlags Flags);
  std::error_code ChangeStdoutMode(fs::OpenFlags Flags);

}  // namespace sys

}  // namespace domino

#endif  // DOMINO_SUPPORT_PROGRAM_H_