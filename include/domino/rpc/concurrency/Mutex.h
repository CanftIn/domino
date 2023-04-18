#ifndef DOMINO_RPC_CONCURRENCY_MUTEX_H_
#define DOMINO_RPC_CONCURRENCY_MUTEX_H_

#include <domino/util/NonCopyable.h>

#include <cstdint>
#include <memory>

namespace domino {
namespace rpc {
namespace concurrency {

class Mutex {
 public:
  Mutex();
  virtual ~Mutex() = default;

  virtual void lock() const;
  virtual void unlock() const;
  virtual bool trylock() const;
  virtual bool timelock(uint64_t millisecond) const;

  void* getUnderlyingImpl() const;

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};

class MutexGuard : public NonCopyable {
 public:
  explicit MutexGuard(const Mutex& mutex, uint64_t timeout = 0)
      : mutex_(&mutex) {
    if (timeout == 0) {
      mutex.lock();
    } else if (timeout < 0) {
      if (!mutex.trylock()) {
        mutex_ = nullptr;
      }
    } else {
      if (!mutex.timelock(timeout)) {
        mutex_ = nullptr;
      }
    }
  }
  ~MutexGuard() {
    if (mutex_) {
      mutex_->unlock();
    }
  }

 private:
  const Mutex* mutex_;
};

}  // namespace concurrency
}  // namespace rpc
}  // namespace domino

#endif  // DOMINO_RPC_CONCURRENCY_MUTEX_H_