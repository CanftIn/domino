#include <domino/support/filesystem/Errno.h>
#include <gtest/gtest.h>

using namespace domino::sys;

TEST(ErrnoTest, RetryAfterSignal) {
  EXPECT_EQ(1, RetryAfterSignal(-1, [] { return 1; }));

  EXPECT_EQ(-1, RetryAfterSignal(-1, [] {
    errno = EAGAIN;
    return -1;
  }));
  EXPECT_EQ(EAGAIN, errno);

  unsigned calls = 0;
  EXPECT_EQ(1, RetryAfterSignal(-1, [&calls] {
              errno = EINTR;
              ++calls;
              return calls == 1 ? -1 : 1;
            }));
  EXPECT_EQ(2u, calls);

  EXPECT_EQ(1, RetryAfterSignal(
                   -1, [](int x) { return x; }, 1));

  std::unique_ptr<int> P(RetryAfterSignal(nullptr, [] { return new int(47); }));
  EXPECT_EQ(47, *P);

  errno = EINTR;
  EXPECT_EQ(-1, RetryAfterSignal(-1, [] { return -1; }));
}
