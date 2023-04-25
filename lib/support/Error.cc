#include <domino/support/Error.h>

namespace domino {

Error errorCodeToError(std::error_code EC) {
  if (!EC) return Error::success();
  return Error(std::make_unique<ECError>(ECError(EC)));
}

}  // namespace domino