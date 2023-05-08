#include <domino/rpc/concurrency/Mutex.h>
#include <gtest/gtest.h>

using namespace domino::rpc::concurrency;

TEST(MUTEX, Base) {
  Mutex mutex;

  mutex.lock();
  mutex.unlock();

  mutex.lock();
  mutex.unlock();
}