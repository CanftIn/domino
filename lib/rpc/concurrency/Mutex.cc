#include <domino/rpc/concurrency/Mutex.h>

#include <chrono>
#include <mutex>

namespace domino {
namespace rpc {
namespace concurrency {

class Mutex::Impl : public std::timed_mutex {};

Mutex::Mutex() : impl_(new Mutex::Impl()) {}

void* Mutex::getUnderlyingImpl() const { return impl_.get(); }

void Mutex::lock() const { impl_->lock(); }

bool Mutex::trylock() const { return impl_->try_lock(); }

bool Mutex::timelock(uint64_t millisecond) const {
  return impl_->try_lock_for(std::chrono::milliseconds(millisecond));
}

void Mutex::unlock() const { impl_->unlock(); }

}  // namespace concurrency
}  // namespace rpc
}  // namespace domino