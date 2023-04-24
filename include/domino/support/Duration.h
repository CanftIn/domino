#ifndef DOMINO_SUPPORT_DURATION_H_
#define DOMINO_SUPPORT_DURATION_H_

#include <chrono>

namespace domino {

class Duration {
  std::chrono::milliseconds Value;

 public:
  Duration(std::chrono::milliseconds Value) : Value(Value) {}
  std::chrono::milliseconds getDuration() const { return Value; }
};

}  // namespace domino

#endif  // DOMINO_SUPPORT_DURATION_H_