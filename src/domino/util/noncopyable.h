#ifndef DOMINO_NONCOPYABLE_H_
#define DOMINO_NONCOPYABLE_H_

namespace domino {

class NonCopyable {
 protected:
  NonCopyable() = default;
  ~NonCopyable() = default;

  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

}  // namespace domino

#endif  // DOMINO_NONCOPYABLE_H_