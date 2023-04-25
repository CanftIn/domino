#include <domino/support/Program.h>

namespace domino {

namespace sys {

std::error_code ChangeStdinToBinary() {
  // Do nothing, as Unix doesn't differentiate between text and binary.
  return std::error_code();
}

std::error_code ChangeStdoutToBinary() {
  // Do nothing, as Unix doesn't differentiate between text and binary.
  return std::error_code();
}

std::error_code ChangeStdinMode(fs::OpenFlags Flags) {
  if (!(Flags & fs::OF_Text)) return ChangeStdinToBinary();
  return std::error_code();
}

std::error_code ChangeStdoutMode(fs::OpenFlags Flags) {
  if (!(Flags & fs::OF_Text)) return ChangeStdoutToBinary();
  return std::error_code();
}

}  // namespace sys

}  // namespace domino