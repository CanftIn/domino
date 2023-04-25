#include <domino/support/Process.h>
#include <domino/support/filesystem/Unix.h>
#include <domino/util/Hashing.h>
#include <domino/util/StringRef.h>
#include <fcntl.h>
#include <malloc.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <termios.h>

#include <mutex>
#include <optional>

using namespace domino;
using namespace sys;

static std::pair<std::chrono::microseconds, std::chrono::microseconds>
getRUsageTimes() {
  struct rusage RU;
  ::getrusage(RUSAGE_SELF, &RU);
  return {toDuration(RU.ru_utime), toDuration(RU.ru_stime)};
}

Process::Pid Process::getProcessId() {
  static_assert(sizeof(Pid) >= sizeof(pid_t),
                "Process::Pid should be big enough to store pid_t");
  return Pid(::getpid());
}

Expected<unsigned> Process::getPageSize() {
  static const int page_size = ::getpagesize();

  if (page_size == -1)
    return errorCodeToError(std::error_code(errno, std::generic_category()));

  return static_cast<unsigned>(page_size);
}

size_t Process::GetMallocUsage() {
  struct mallinfo2 mi;
  mi = ::mallinfo2();
  return mi.uordblks;
}

void Process::GetTimeUsage(TimePoint<> &elapsed,
                           std::chrono::nanoseconds &user_time,
                           std::chrono::nanoseconds &sys_time) {
  elapsed = std::chrono::system_clock::now();
  std::tie(user_time, sys_time) = getRUsageTimes();
}

void Process::PreventCoreFiles() {
  struct rlimit rlim;
  rlim.rlim_cur = rlim.rlim_max = 0;
  setrlimit(RLIMIT_CORE, &rlim);

  coreFilesPrevented = true;
}

std::optional<std::string> Process::GetEnv(StringRef Name) {
  std::string NameStr = Name.str();
  const char *Val = ::getenv(NameStr.c_str());
  if (!Val) return std::nullopt;
  return std::string(Val);
}

namespace {
class FDCloser {
 public:
  FDCloser(int &FD) : FD(FD), KeepOpen(false) {}
  void keepOpen() { KeepOpen = true; }
  ~FDCloser() {
    if (!KeepOpen && FD >= 0) ::close(FD);
  }

 private:
  FDCloser(const FDCloser &) = delete;
  void operator=(const FDCloser &) = delete;

  int &FD;
  bool KeepOpen;
};
}  // namespace

std::error_code Process::FixupStandardFileDescriptors() {
  int NullFD = -1;
  FDCloser FDC(NullFD);
  const int StandardFDs[] = {STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO};
  for (int StandardFD : StandardFDs) {
    struct stat st;
    errno = 0;
    if (RetryAfterSignal(-1, ::fstat, StandardFD, &st) < 0) {
      assert(errno && "expected errno to be set if fstat failed!");
      // fstat should return EBADF if the file descriptor is closed.
      if (errno != EBADF)
        return std::error_code(errno, std::generic_category());
    }
    // if fstat succeeds, move on to the next FD.
    if (!errno) continue;
    assert(errno == EBADF && "expected errno to have EBADF at this point!");

    if (NullFD < 0) {
      // Call ::open in a lambda to avoid overload resolution in
      // RetryAfterSignal when open is overloaded, such as in Bionic.
      auto Open = [&]() { return ::open("/dev/null", O_RDWR); };
      if ((NullFD = RetryAfterSignal(-1, Open)) < 0)
        return std::error_code(errno, std::generic_category());
    }

    if (NullFD == StandardFD)
      FDC.keepOpen();
    else if (dup2(NullFD, StandardFD) < 0)
      return std::error_code(errno, std::generic_category());
  }
  return std::error_code();
}

std::error_code Process::SafelyCloseFileDescriptor(int FD) {
  // Create a signal set filled with *all* signals.
  sigset_t FullSet, SavedSet;
  if (sigfillset(&FullSet) < 0 || sigfillset(&SavedSet) < 0)
    return std::error_code(errno, std::generic_category());

    // Atomically swap our current signal mask with a full mask.
#if DOMINO_ENABLE_THREADS
  if (int EC = pthread_sigmask(SIG_SETMASK, &FullSet, &SavedSet))
    return std::error_code(EC, std::generic_category());
#else
  if (sigprocmask(SIG_SETMASK, &FullSet, &SavedSet) < 0)
    return std::error_code(errno, std::generic_category());
#endif
  // Attempt to close the file descriptor.
  // We need to save the error, if one occurs, because our subsequent call to
  // pthread_sigmask might tamper with errno.
  int ErrnoFromClose = 0;
  if (::close(FD) < 0) ErrnoFromClose = errno;
  // Restore the signal mask back to what we saved earlier.
  int EC = 0;
#if DOMINO_ENABLE_THREADS
  EC = pthread_sigmask(SIG_SETMASK, &SavedSet, nullptr);
#else
  if (sigprocmask(SIG_SETMASK, &SavedSet, nullptr) < 0) EC = errno;
#endif
  // The error code from close takes precedence over the one from
  // pthread_sigmask.
  if (ErrnoFromClose)
    return std::error_code(ErrnoFromClose, std::generic_category());
  return std::error_code(EC, std::generic_category());
}

bool Process::StandardInIsUserInput() {
  return FileDescriptorIsDisplayed(STDIN_FILENO);
}

bool Process::StandardOutIsDisplayed() {
  return FileDescriptorIsDisplayed(STDOUT_FILENO);
}

bool Process::StandardErrIsDisplayed() {
  return FileDescriptorIsDisplayed(STDERR_FILENO);
}

bool Process::FileDescriptorIsDisplayed(int fd) {
#if HAVE_ISATTY
  return isatty(fd);
#else
  // If we don't have isatty, just return false.
  return false;
#endif
}

static unsigned getColumns() {
  // If COLUMNS is defined in the environment, wrap to that many columns.
  if (const char *ColumnsStr = std::getenv("COLUMNS")) {
    int Columns = std::atoi(ColumnsStr);
    if (Columns > 0) return Columns;
  }

  // We used to call ioctl TIOCGWINSZ to determine the width. It is considered
  // unuseful.
  return 0;
}

unsigned Process::StandardOutColumns() {
  if (!StandardOutIsDisplayed()) return 0;

  return getColumns();
}

unsigned Process::StandardErrColumns() {
  if (!StandardErrIsDisplayed()) return 0;

  return getColumns();
}

#ifdef DOMINO_ENABLE_TERMINFO
// We manually declare these extern functions because finding the correct
// headers from various terminfo, curses, or other sources is harder than
// writing their specs down.
extern "C" int setupterm(char *term, int filedes, int *errret);
extern "C" struct term *set_curterm(struct term *termp);
extern "C" int del_curterm(struct term *termp);
extern "C" int tigetnum(char *capname);
#endif

bool checkTerminalEnvironmentForColors() {
  if (const char *TermStr = std::getenv("TERM")) {
    return StringSwitch<bool>(TermStr)
        .Case("ansi", true)
        .Case("cygwin", true)
        .Case("linux", true)
        .StartsWith("screen", true)
        .StartsWith("xterm", true)
        .StartsWith("vt100", true)
        .StartsWith("rxvt", true)
        .EndsWith("color", true)
        .Default(false);
  }

  return false;
}

static bool terminalHasColors(int fd) {
#ifdef DOMINO_ENABLE_TERMINFO
  // First, acquire a global lock because these C routines are thread hostile.
  static std::mutex TermColorMutex;
  std::lock_guard<std::mutex> G(TermColorMutex);

  struct term *previous_term = set_curterm(nullptr);
  int errret = 0;
  if (setupterm(nullptr, fd, &errret) != 0)
    // Regardless of why, if we can't get terminfo, we shouldn't try to print
    // colors.
    return false;

  // Test whether the terminal as set up supports color output. How to do this
  // isn't entirely obvious. We can use the curses routine 'has_colors' but it
  // would be nice to avoid a dependency on curses proper when we can make do
  // with a minimal terminfo parsing library. Also, we don't really care whether
  // the terminal supports the curses-specific color changing routines, merely
  // if it will interpret ANSI color escape codes in a reasonable way. Thus, the
  // strategy here is just to query the baseline colors capability and if it
  // supports colors at all to assume it will translate the escape codes into
  // whatever range of colors it does support. We can add more detailed tests
  // here if users report them as necessary.
  //
  // The 'tigetnum' routine returns -2 or -1 on errors, and might return 0 if
  // the terminfo says that no colors are supported.
  int colors_ti = tigetnum(const_cast<char *>("colors"));
  bool HasColors =
      colors_ti >= 0 ? colors_ti : checkTerminalEnvironmentForColors();

  // Now extract the structure allocated by setupterm and free its memory
  // through a really silly dance.
  struct term *termp = set_curterm(previous_term);
  (void)del_curterm(termp);  // Drop any errors here.

  // Return true if we found a color capabilities for the current terminal.
  return HasColors;
#else
  // When the terminfo database is not available, check if the current terminal
  // is one of terminals that are known to support ANSI color escape codes.
  return checkTerminalEnvironmentForColors();
#endif
}

bool Process::FileDescriptorHasColors(int fd) {
  // A file descriptor has colors if it is displayed and the terminal has
  // colors.
  return FileDescriptorIsDisplayed(fd) && terminalHasColors(fd);
}

bool Process::StandardOutHasColors() {
  return FileDescriptorHasColors(STDOUT_FILENO);
}

bool Process::StandardErrHasColors() {
  return FileDescriptorHasColors(STDERR_FILENO);
}

void Process::UseANSIEscapeCodes(bool /*enable*/) {
  // No effect.
}

bool Process::ColorNeedsFlush() {
  // No, we use ANSI escape sequences.
  return false;
}

const char *Process::OutputColor(char code, bool bold, bool bg) {
  return colorcodes[bg ? 1 : 0][bold ? 1 : 0][code & 7];
}

const char *Process::OutputBold(bool bg) { return "\033[1m"; }

const char *Process::OutputReverse() { return "\033[7m"; }

const char *Process::ResetColor() { return "\033[0m"; }

#if !HAVE_DECL_ARC4RANDOM
static unsigned GetRandomNumberSeed() {
  // Attempt to get the initial seed from /dev/urandom, if possible.
  int urandomFD = open("/dev/urandom", O_RDONLY);

  if (urandomFD != -1) {
    unsigned seed;
    // Don't use a buffered read to avoid reading more data
    // from /dev/urandom than we need.
    int count = read(urandomFD, (void *)&seed, sizeof(seed));

    close(urandomFD);

    // Return the seed if the read was successful.
    if (count == sizeof(seed)) return seed;
  }

  // Otherwise, swizzle the current time and the process ID to form a reasonable
  // seed.
  const auto Now = std::chrono::high_resolution_clock::now();
  return hash_combine(Now.time_since_epoch().count(), ::getpid());
}
#endif

unsigned domino::sys::Process::GetRandomNumber() {
#if HAVE_DECL_ARC4RANDOM
  return arc4random();
#else
  static int x = (static_cast<void>(::srand(GetRandomNumberSeed())), 0);
  (void)x;
  return ::rand();
#endif
}

[[noreturn]] void Process::ExitNoCleanup(int RetCode) { _Exit(RetCode); }
