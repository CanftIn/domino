#ifndef DOMINO_SUPPORT_FILESYSTEM_PATH_INC_
#define DOMINO_SUPPORT_FILESYSTEM_PATH_INC_

#include <dirent.h>
#include <domino/support/Errc.h>
#include <domino/support/FileSystem.h>
#include <domino/support/Process.h>
#include <domino/support/filesystem/Path.h>
#include <domino/support/filesystem/Unix.h>
#include <domino/util/Logging.h>
#include <fcntl.h>
#include <limits.h>
#include <pwd.h>
#include <stdio.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Both stdio.h and cstdio are included via different paths and
// stdcxx's cstdio doesn't include stdio.h, so it doesn't #undef the macros
// either.
#undef ferror
#undef feof

#include <sys/types.h>
#include <sys/vfs.h>

#define STATVFS statfs
#define FSTATVFS fstatfs
#define STATVFS_F_FRSIZE(vfs) static_cast<uint64_t>(vfs.f_bsize)

#define STATVFS_F_FLAG(vfs) (vfs).f_flags

using namespace domino;

namespace domino {
namespace sys {
namespace fs {

const file_t kInvalidFile = -1;

static int test_dir(char ret[PATH_MAX], const char *dir, const char *bin) {
  struct stat sb;
  char fullpath[PATH_MAX];

  int chars = snprintf(fullpath, PATH_MAX, "%s/%s", dir, bin);
  // We cannot write PATH_MAX characters because the string will be terminated
  // with a null character. Fail if truncation happened.
  if (chars >= PATH_MAX) return 1;
  if (!realpath(fullpath, ret)) return 1;
  if (stat(fullpath, &sb) != 0) return 1;

  return 0;
}

static char *getprogpath(char ret[PATH_MAX], const char *bin) {
  if (bin == nullptr) return nullptr;

  /* First approach: absolute path. */
  if (bin[0] == '/') {
    if (test_dir(ret, "/", bin) == 0) return ret;
    return nullptr;
  }

  /* Second approach: relative path. */
  if (strchr(bin, '/')) {
    char cwd[PATH_MAX];
    if (!getcwd(cwd, PATH_MAX)) return nullptr;
    if (test_dir(ret, cwd, bin) == 0) return ret;
    return nullptr;
  }

  /* Third approach: $PATH */
  char *pv;
  if ((pv = getenv("PATH")) == nullptr) return nullptr;
  char *s = strdup(pv);
  if (!s) return nullptr;
  char *state;
  for (char *t = strtok_r(s, ":", &state); t != nullptr;
       t = strtok_r(nullptr, ":", &state)) {
    if (test_dir(ret, t, bin) == 0) {
      free(s);
      return ret;
    }
  }
  free(s);
  return nullptr;
}

/// GetMainExecutable - Return the path to the main executable, given the
/// value of argv[0] from program startup.
std::string getMainExecutableImpl(const char *argv0, void *MainAddr) {
  char exe_path[PATH_MAX];
  const char *aPath = "/proc/self/exe";
  if (sys::fs::exists(aPath)) {
    // /proc is not always mounted under Linux (chroot for example).
    ssize_t len = readlink(aPath, exe_path, sizeof(exe_path));
    if (len < 0) return "";

    // Null terminate the string for realpath. readlink never null
    // terminates its output.
    len = std::min(len, ssize_t(sizeof(exe_path) - 1));
    exe_path[len] = '\0';

    // On Linux, /proc/self/exe always looks through symlinks. However, on
    // GNU/Hurd, /proc/self/exe is a symlink to the path that was used to start
    // the program, and not the eventual binary file. Therefore, call realpath
    // so this behaves the same on all platforms.
    if (char *real_path = realpath(exe_path, nullptr)) {
      std::string ret = std::string(real_path);
      free(real_path);
      return ret;
    }
  }
  // Fall back to the classical detection.
  if (getprogpath(exe_path, argv0)) return exe_path;
  return "";
}

TimePoint<> basic_file_status::getLastAccessedTime() const {
  return toTimePoint(fs_st_atime, fs_st_atime_nsec);
}

TimePoint<> basic_file_status::getLastModificationTime() const {
  return toTimePoint(fs_st_mtime, fs_st_mtime_nsec);
}

UniqueID file_status::getUniqueID() const {
  return UniqueID(fs_st_dev, fs_st_ino);
}

uint32_t file_status::getLinkCount() const { return fs_st_nlinks; }

ErrorOr<space_info> disk_space(const Twine &Path) {
  struct STATVFS Vfs;
  if (::STATVFS(const_cast<char *>(Path.str().c_str()), &Vfs))
    return std::error_code(errno, std::generic_category());
  auto FrSize = STATVFS_F_FRSIZE(Vfs);
  space_info SpaceInfo;
  SpaceInfo.capacity = static_cast<uint64_t>(Vfs.f_blocks) * FrSize;
  SpaceInfo.free = static_cast<uint64_t>(Vfs.f_bfree) * FrSize;
  SpaceInfo.available = static_cast<uint64_t>(Vfs.f_bavail) * FrSize;
  return SpaceInfo;
}

std::error_code current_path(SmallVectorImpl<char> &result) {
  result.clear();

  const char *pwd = ::getenv("PWD");
  domino::sys::fs::file_status PWDStatus, DotStatus;
  if (pwd && domino::sys::path::is_absolute(pwd) &&
      !domino::sys::fs::status(pwd, PWDStatus) &&
      !domino::sys::fs::status(".", DotStatus) &&
      PWDStatus.getUniqueID() == DotStatus.getUniqueID()) {
    result.append(pwd, pwd + strlen(pwd));
    return std::error_code();
  }

  result.resize_for_overwrite(PATH_MAX);

  while (true) {
    if (::getcwd(result.data(), result.size()) == nullptr) {
      // See if there was a real error.
      if (errno != ENOMEM) {
        result.clear();
        return std::error_code(errno, std::generic_category());
      }
      // Otherwise there just wasn't enough space.
      result.resize_for_overwrite(result.capacity() * 2);
    } else
      break;
  }

  result.truncate(strlen(result.data()));
  return std::error_code();
}

std::error_code set_current_path(const Twine &path) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::chdir(p.begin()) == -1)
    return std::error_code(errno, std::generic_category());

  return std::error_code();
}

std::error_code create_directory(const Twine &path, bool IgnoreExisting,
                                 perms Perms) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::mkdir(p.begin(), Perms) == -1) {
    if (errno != EEXIST || !IgnoreExisting)
      return std::error_code(errno, std::generic_category());
  }

  return std::error_code();
}

// Note that we are using symbolic link because hard links are not supported by
// all filesystems (SMB doesn't).
std::error_code create_link(const Twine &to, const Twine &from) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::symlink(t.begin(), f.begin()) == -1)
    return std::error_code(errno, std::generic_category());

  return std::error_code();
}

std::error_code create_hard_link(const Twine &to, const Twine &from) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::link(t.begin(), f.begin()) == -1)
    return std::error_code(errno, std::generic_category());

  return std::error_code();
}

std::error_code remove(const Twine &path, bool IgnoreNonExisting) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  struct stat buf;
  if (lstat(p.begin(), &buf) != 0) {
    if (errno != ENOENT || !IgnoreNonExisting)
      return std::error_code(errno, std::generic_category());
    return std::error_code();
  }

  // Note: this check catches strange situations. In all cases, LLVM should
  // only be involved in the creation and deletion of regular files.  This
  // check ensures that what we're trying to erase is a regular file. It
  // effectively prevents LLVM from erasing things like /dev/null, any block
  // special file, or other things that aren't "regular" files.
  if (!S_ISREG(buf.st_mode) && !S_ISDIR(buf.st_mode) && !S_ISLNK(buf.st_mode))
    return make_error_code(errc::operation_not_permitted);

  if (::remove(p.begin()) == -1) {
    if (errno != ENOENT || !IgnoreNonExisting)
      return std::error_code(errno, std::generic_category());
  }

  return std::error_code();
}

static bool is_local_impl(struct STATVFS &Vfs) {
#ifndef NFS_SUPER_MAGIC
#define NFS_SUPER_MAGIC 0x6969
#endif
#ifndef SMB_SUPER_MAGIC
#define SMB_SUPER_MAGIC 0x517B
#endif
#ifndef CIFS_MAGIC_NUMBER
#define CIFS_MAGIC_NUMBER 0xFF534D42
#endif

  switch ((uint32_t)Vfs.f_type) {
    case NFS_SUPER_MAGIC:
    case SMB_SUPER_MAGIC:
    case CIFS_MAGIC_NUMBER:
      return false;
    default:
      return true;
  }
}

std::error_code is_local(const Twine &Path, bool &Result) {
  struct STATVFS Vfs;
  if (::STATVFS(const_cast<char *>(Path.str().c_str()), &Vfs))
    return std::error_code(errno, std::generic_category());

  Result = is_local_impl(Vfs);
  return std::error_code();
}

std::error_code is_local(int FD, bool &Result) {
  struct STATVFS Vfs;
  if (::FSTATVFS(FD, &Vfs))
    return std::error_code(errno, std::generic_category());

  Result = is_local_impl(Vfs);
  return std::error_code();
}

std::error_code rename(const Twine &from, const Twine &to) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::rename(f.begin(), t.begin()) == -1)
    return std::error_code(errno, std::generic_category());

  return std::error_code();
}

std::error_code resize_file(int FD, uint64_t Size) {
  // Use ftruncate as a fallback. It may or may not allocate space. At least on
  // OS X with HFS+ it does.
  if (::ftruncate(FD, Size) == -1)
    return std::error_code(errno, std::generic_category());

  return std::error_code();
}

static int convertAccessMode(AccessMode Mode) {
  switch (Mode) {
    case AccessMode::Exist:
      return F_OK;
    case AccessMode::Write:
      return W_OK;
    case AccessMode::Execute:
      return R_OK | X_OK;  // scripts also need R_OK.
  }
  DOMINO_ERROR_ABORT("invalid enum");
}

std::error_code access(const Twine &Path, AccessMode Mode) {
  SmallString<128> PathStorage;
  StringRef P = Path.toNullTerminatedStringRef(PathStorage);

  if (::access(P.begin(), convertAccessMode(Mode)) == -1)
    return std::error_code(errno, std::generic_category());

  if (Mode == AccessMode::Execute) {
    // Don't say that directories are executable.
    struct stat buf;
    if (0 != stat(P.begin(), &buf)) return errc::permission_denied;
    if (!S_ISREG(buf.st_mode)) return errc::permission_denied;
  }

  return std::error_code();
}

bool can_execute(const Twine &Path) {
  return !access(Path, AccessMode::Execute);
}

bool equivalent(file_status A, file_status B) {
  assert(status_known(A) && status_known(B));
  return A.fs_st_dev == B.fs_st_dev && A.fs_st_ino == B.fs_st_ino;
}

std::error_code equivalent(const Twine &A, const Twine &B, bool &result) {
  file_status fsA, fsB;
  if (std::error_code ec = status(A, fsA)) return ec;
  if (std::error_code ec = status(B, fsB)) return ec;
  result = equivalent(fsA, fsB);
  return std::error_code();
}

static void expandTildeExpr(SmallVectorImpl<char> &Path) {
  StringRef PathStr(Path.begin(), Path.size());
  if (PathStr.empty() || !PathStr.starts_with("~")) return;

  PathStr = PathStr.drop_front();
  StringRef Expr =
      PathStr.take_until([](char c) { return path::is_separator(c); });
  StringRef Remainder = PathStr.substr(Expr.size() + 1);
  SmallString<128> Storage;
  if (Expr.empty()) {
    // This is just ~/..., resolve it to the current user's home dir.
    if (!path::home_directory(Storage)) {
      // For some reason we couldn't get the home directory.  Just exit.
      return;
    }

    // Overwrite the first character and insert the rest.
    Path[0] = Storage[0];
    Path.insert(Path.begin() + 1, Storage.begin() + 1, Storage.end());
    return;
  }

  // This is a string of the form ~username/, look up this user's entry in the
  // password database.
  std::unique_ptr<char[]> Buf;
  long BufSize = sysconf(_SC_GETPW_R_SIZE_MAX);
  if (BufSize <= 0) BufSize = 16384;
  Buf = std::make_unique<char[]>(BufSize);
  struct passwd Pwd;
  std::string User = Expr.str();
  struct passwd *Entry = nullptr;
  getpwnam_r(User.c_str(), &Pwd, Buf.get(), BufSize, &Entry);

  if (!Entry || !Entry->pw_dir) {
    // Unable to look up the entry, just return back the original path.
    return;
  }

  Storage = Remainder;
  Path.clear();
  Path.append(Entry->pw_dir, Entry->pw_dir + strlen(Entry->pw_dir));
  domino::sys::path::append(Path, Storage);
}

void expand_tilde(const Twine &path, SmallVectorImpl<char> &dest) {
  dest.clear();
  if (path.isTriviallyEmpty()) return;

  path.toVector(dest);
  expandTildeExpr(dest);
}

static file_type typeForMode(mode_t Mode) {
  if (S_ISDIR(Mode))
    return file_type::directory_file;
  else if (S_ISREG(Mode))
    return file_type::regular_file;
  else if (S_ISBLK(Mode))
    return file_type::block_file;
  else if (S_ISCHR(Mode))
    return file_type::character_file;
  else if (S_ISFIFO(Mode))
    return file_type::fifo_file;
  else if (S_ISSOCK(Mode))
    return file_type::socket_file;
  else if (S_ISLNK(Mode))
    return file_type::symlink_file;
  return file_type::type_unknown;
}

static std::error_code fillStatus(int StatRet, const struct stat &Status,
                                  file_status &Result) {
  if (StatRet != 0) {
    std::error_code EC(errno, std::generic_category());
    if (EC == errc::no_such_file_or_directory)
      Result = file_status(file_type::file_not_found);
    else
      Result = file_status(file_type::status_error);
    return EC;
  }

  uint32_t atime_nsec, mtime_nsec;

  atime_nsec = mtime_nsec = 0;

  perms Perms = static_cast<perms>(Status.st_mode) & all_perms;
  Result = file_status(typeForMode(Status.st_mode), Perms, Status.st_dev,
                       Status.st_nlink, Status.st_ino, Status.st_atime,
                       atime_nsec, Status.st_mtime, mtime_nsec, Status.st_uid,
                       Status.st_gid, Status.st_size);

  return std::error_code();
}

std::error_code status(const Twine &Path, file_status &Result, bool Follow) {
  SmallString<128> PathStorage;
  StringRef P = Path.toNullTerminatedStringRef(PathStorage);

  struct stat Status;
  int StatRet = (Follow ? ::stat : ::lstat)(P.begin(), &Status);
  return fillStatus(StatRet, Status, Result);
}

std::error_code status(int FD, file_status &Result) {
  struct stat Status;
  int StatRet = ::fstat(FD, &Status);
  return fillStatus(StatRet, Status, Result);
}

unsigned getUmask() {
  // Chose arbitary new mask and reset the umask to the old mask.
  // umask(2) never fails so ignore the return of the second call.
  unsigned Mask = ::umask(0);
  (void)::umask(Mask);
  return Mask;
}

std::error_code setPermissions(const Twine &Path, perms Permissions) {
  SmallString<128> PathStorage;
  StringRef P = Path.toNullTerminatedStringRef(PathStorage);

  if (::chmod(P.begin(), Permissions))
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

std::error_code setPermissions(int FD, perms Permissions) {
  if (::fchmod(FD, Permissions))
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

std::error_code setLastAccessAndModificationTime(int FD, TimePoint<> AccessTime,
                                                 TimePoint<> ModificationTime) {
  timespec Times[2];
  Times[0] = sys::toTimeSpec(AccessTime);
  Times[1] = sys::toTimeSpec(ModificationTime);
  if (::futimens(FD, Times))
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

std::error_code mapped_file_region::init(int FD, uint64_t Offset,
                                         mapmode Mode) {
  assert(Size != 0);

  int flags = (Mode == readwrite) ? MAP_SHARED : MAP_PRIVATE;
  int prot = (Mode == readonly) ? PROT_READ : (PROT_READ | PROT_WRITE);

  flags |= MAP_NORESERVE;

  Mapping = ::mmap(nullptr, Size, prot, flags, FD, Offset);
  if (Mapping == MAP_FAILED)
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

mapped_file_region::mapped_file_region(int fd, mapmode mode, size_t length,
                                       uint64_t offset, std::error_code &ec)
    : Size(length), Mode(mode) {
  (void)Mode;
  ec = init(fd, offset, mode);
  if (ec) copyFrom(mapped_file_region());
}

void mapped_file_region::unmapImpl() {
  if (Mapping) ::munmap(Mapping, Size);
}

void mapped_file_region::dontNeedImpl() {
  assert(Mode == mapped_file_region::readonly);
  if (!Mapping) return;
  ::posix_madvise(Mapping, Size, POSIX_MADV_DONTNEED);
}

int mapped_file_region::alignment() { return Process::getPageSizeEstimate(); }

std::error_code detail::directory_iterator_construct(detail::DirIterState &it,
                                                     StringRef path,
                                                     bool follow_symlinks) {
  SmallString<128> path_null(path);
  DIR *directory = ::opendir(path_null.c_str());
  if (!directory) return std::error_code(errno, std::generic_category());

  it.IterationHandle = reinterpret_cast<intptr_t>(directory);
  // Add something for replace_filename to replace.
  path::append(path_null, ".");
  it.CurrentEntry = directory_entry(path_null.str(), follow_symlinks);
  return directory_iterator_increment(it);
}

std::error_code detail::directory_iterator_destruct(detail::DirIterState &it) {
  if (it.IterationHandle)
    ::closedir(reinterpret_cast<DIR *>(it.IterationHandle));
  it.IterationHandle = 0;
  it.CurrentEntry = directory_entry();
  return std::error_code();
}

static file_type direntType(dirent *Entry) {
  // Most platforms provide the file type in the dirent: Linux/BSD/Mac.
  // The DTTOIF macro lets us reuse our status -> type conversion.
  // Note that while glibc provides a macro to see if this is supported,
  // _DIRENT_HAVE_D_TYPE, it's not defined on BSD/Mac, so we test for the
  // d_type-to-mode_t conversion macro instead.
  return typeForMode(DTTOIF(Entry->d_type));
}

std::error_code detail::directory_iterator_increment(detail::DirIterState &It) {
  errno = 0;
  dirent *CurDir = ::readdir(reinterpret_cast<DIR *>(It.IterationHandle));
  if (CurDir == nullptr && errno != 0) {
    return std::error_code(errno, std::generic_category());
  } else if (CurDir != nullptr) {
    StringRef Name(CurDir->d_name);
    if ((Name.size() == 1 && Name[0] == '.') ||
        (Name.size() == 2 && Name[0] == '.' && Name[1] == '.'))
      return directory_iterator_increment(It);
    It.CurrentEntry.replace_filename(Name, direntType(CurDir));
  } else
    return directory_iterator_destruct(It);

  return std::error_code();
}

ErrorOr<basic_file_status> directory_entry::status() const {
  file_status s;
  if (auto EC = fs::status(Path, s, FollowSymlinks)) return EC;
  return s;
}

//
// FreeBSD optionally provides /proc/self/fd, but it is incompatible with
// Linux. The thing to use is realpath.
//
#if !defined(__FreeBSD__)
#define TRY_PROC_SELF_FD
#endif

#if !defined(F_GETPATH) && defined(TRY_PROC_SELF_FD)
static bool hasProcSelfFD() {
  // If we have a /proc filesystem mounted, we can quickly establish the
  // real name of the file with readlink
  static const bool Result = (::access("/proc/self/fd", R_OK) == 0);
  return Result;
}
#endif

static int nativeOpenFlags(CreationDisposition Disp, OpenFlags Flags,
                           FileAccess Access) {
  int Result = 0;
  if (Access == FA_Read)
    Result |= O_RDONLY;
  else if (Access == FA_Write)
    Result |= O_WRONLY;
  else if (Access == (FA_Read | FA_Write))
    Result |= O_RDWR;

  // This is for compatibility with old code that assumed OF_Append implied
  // would open an existing file.  See Windows/Path.inc for a longer comment.
  if (Flags & OF_Append) Disp = CD_OpenAlways;

  if (Disp == CD_CreateNew) {
    Result |= O_CREAT;  // Create if it doesn't exist.
    Result |= O_EXCL;   // Fail if it does.
  } else if (Disp == CD_CreateAlways) {
    Result |= O_CREAT;  // Create if it doesn't exist.
    Result |= O_TRUNC;  // Truncate if it does.
  } else if (Disp == CD_OpenAlways) {
    Result |= O_CREAT;  // Create if it doesn't exist.
  } else if (Disp == CD_OpenExisting) {
    // Nothing special, just don't add O_CREAT and we get these semantics.
  }

// Using append mode with z/OS UTF-8 auto-conversion results in EINVAL when
// calling write(). Instead we need to use lseek() to set offset to EOF after
// open().
#ifndef __MVS__
  if (Flags & OF_Append) Result |= O_APPEND;
#endif

#ifdef O_CLOEXEC
  if (!(Flags & OF_ChildInherit)) Result |= O_CLOEXEC;
#endif

  return Result;
}

std::error_code openFile(const Twine &Name, int &ResultFD,
                         CreationDisposition Disp, FileAccess Access,
                         OpenFlags Flags, unsigned Mode) {
  int OpenFlags = nativeOpenFlags(Disp, Flags, Access);

  SmallString<128> Storage;
  StringRef P = Name.toNullTerminatedStringRef(Storage);
  // Call ::open in a lambda to avoid overload resolution in RetryAfterSignal
  // when open is overloaded, such as in Bionic.
  auto Open = [&]() { return ::open(P.begin(), OpenFlags, Mode); };
  if ((ResultFD = sys::RetryAfterSignal(-1, Open)) < 0)
    return std::error_code(errno, std::generic_category());

  return std::error_code();
}

Expected<int> openNativeFile(const Twine &Name, CreationDisposition Disp,
                             FileAccess Access, OpenFlags Flags,
                             unsigned Mode) {
  int FD;
  std::error_code EC = openFile(Name, FD, Disp, Access, Flags, Mode);
  if (EC) return errorCodeToError(EC);
  return FD;
}

std::error_code openFileForRead(const Twine &Name, int &ResultFD,
                                OpenFlags Flags,
                                SmallVectorImpl<char> *RealPath) {
  std::error_code EC =
      openFile(Name, ResultFD, CD_OpenExisting, FA_Read, Flags, 0666);
  if (EC) return EC;

  // Attempt to get the real name of the file, if the user asked
  if (!RealPath) return std::error_code();
  RealPath->clear();
#if defined(F_GETPATH)
  // When F_GETPATH is availble, it is the quickest way to get
  // the real path name.
  char Buffer[PATH_MAX];
  if (::fcntl(ResultFD, F_GETPATH, Buffer) != -1)
    RealPath->append(Buffer, Buffer + strlen(Buffer));
#else
  char Buffer[PATH_MAX];
#if defined(TRY_PROC_SELF_FD)
  if (hasProcSelfFD()) {
    char ProcPath[64];
    snprintf(ProcPath, sizeof(ProcPath), "/proc/self/fd/%d", ResultFD);
    ssize_t CharCount = ::readlink(ProcPath, Buffer, sizeof(Buffer));
    if (CharCount > 0) RealPath->append(Buffer, Buffer + CharCount);
  } else {
#endif
    SmallString<128> Storage;
    StringRef P = Name.toNullTerminatedStringRef(Storage);

    // Use ::realpath to get the real path name
    if (::realpath(P.begin(), Buffer) != nullptr)
      RealPath->append(Buffer, Buffer + strlen(Buffer));
#if defined(TRY_PROC_SELF_FD)
  }
#endif
#endif
  return std::error_code();
}

Expected<file_t> openNativeFileForRead(const Twine &Name, OpenFlags Flags,
                                       SmallVectorImpl<char> *RealPath) {
  file_t ResultFD;
  std::error_code EC = openFileForRead(Name, ResultFD, Flags, RealPath);
  if (EC) return errorCodeToError(EC);
  return ResultFD;
}

file_t getStdinHandle() { return 0; }
file_t getStdoutHandle() { return 1; }
file_t getStderrHandle() { return 2; }

Expected<size_t> readNativeFile(file_t FD, MutableArrayRef<char> Buf) {
  size_t Size = Buf.size();
  ssize_t NumRead = sys::RetryAfterSignal(-1, ::read, FD, Buf.data(), Size);
  if (ssize_t(NumRead) == -1)
    return errorCodeToError(std::error_code(errno, std::generic_category()));
  return NumRead;
}

Expected<size_t> readNativeFileSlice(file_t FD, MutableArrayRef<char> Buf,
                                     uint64_t Offset) {
  size_t Size = Buf.size();

  if (lseek(FD, Offset, SEEK_SET) == -1)
    return errorCodeToError(std::error_code(errno, std::generic_category()));
  ssize_t NumRead = sys::RetryAfterSignal(-1, ::read, FD, Buf.data(), Size);

  if (NumRead == -1)
    return errorCodeToError(std::error_code(errno, std::generic_category()));
  return NumRead;
}

std::error_code tryLockFile(int FD, std::chrono::milliseconds Timeout) {
  auto Start = std::chrono::steady_clock::now();
  auto End = Start + Timeout;
  do {
    struct flock Lock;
    memset(&Lock, 0, sizeof(Lock));
    Lock.l_type = F_WRLCK;
    Lock.l_whence = SEEK_SET;
    Lock.l_start = 0;
    Lock.l_len = 0;
    if (::fcntl(FD, F_SETLK, &Lock) != -1) return std::error_code();
    int Error = errno;
    if (Error != EACCES && Error != EAGAIN)
      return std::error_code(Error, std::generic_category());
    usleep(1000);
  } while (std::chrono::steady_clock::now() < End);
  return make_error_code(errc::no_lock_available);
}

std::error_code lockFile(int FD) {
  struct flock Lock;
  memset(&Lock, 0, sizeof(Lock));
  Lock.l_type = F_WRLCK;
  Lock.l_whence = SEEK_SET;
  Lock.l_start = 0;
  Lock.l_len = 0;
  if (::fcntl(FD, F_SETLKW, &Lock) != -1) return std::error_code();
  int Error = errno;
  return std::error_code(Error, std::generic_category());
}

std::error_code unlockFile(int FD) {
  struct flock Lock;
  Lock.l_type = F_UNLCK;
  Lock.l_whence = SEEK_SET;
  Lock.l_start = 0;
  Lock.l_len = 0;
  if (::fcntl(FD, F_SETLK, &Lock) != -1) return std::error_code();
  return std::error_code(errno, std::generic_category());
}

std::error_code closeFile(file_t &F) {
  file_t TmpF = F;
  F = kInvalidFile;
  return Process::SafelyCloseFileDescriptor(TmpF);
}

template <typename T>
static std::error_code remove_directories_impl(const T &Entry,
                                               bool IgnoreErrors) {
  std::error_code EC;
  directory_iterator Begin(Entry, EC, false);
  directory_iterator End;
  while (Begin != End) {
    auto &Item = *Begin;
    ErrorOr<basic_file_status> st = Item.status();
    if (st) {
      if (is_directory(*st)) {
        EC = remove_directories_impl(Item, IgnoreErrors);
        if (EC && !IgnoreErrors) return EC;
      }

      EC = fs::remove(Item.path(), true);
      if (EC && !IgnoreErrors) return EC;
    } else if (!IgnoreErrors) {
      return st.getError();
    }

    Begin.increment(EC);
    if (EC && !IgnoreErrors) return EC;
  }
  return std::error_code();
}

std::error_code remove_directories(const Twine &path, bool IgnoreErrors) {
  auto EC = remove_directories_impl(path, IgnoreErrors);
  if (EC && !IgnoreErrors) return EC;
  EC = fs::remove(path, true);
  if (EC && !IgnoreErrors) return EC;
  return std::error_code();
}

std::error_code real_path(const Twine &path, SmallVectorImpl<char> &dest,
                          bool expand_tilde) {
  dest.clear();
  if (path.isTriviallyEmpty()) return std::error_code();

  if (expand_tilde) {
    SmallString<128> Storage;
    path.toVector(Storage);
    expandTildeExpr(Storage);
    return real_path(Storage, dest, false);
  }

  SmallString<128> Storage;
  StringRef P = path.toNullTerminatedStringRef(Storage);
  char Buffer[PATH_MAX];
  if (::realpath(P.begin(), Buffer) == nullptr)
    return std::error_code(errno, std::generic_category());
  dest.append(Buffer, Buffer + strlen(Buffer));
  return std::error_code();
}

std::error_code changeFileOwnership(int FD, uint32_t Owner, uint32_t Group) {
  auto FChown = [&]() { return ::fchown(FD, Owner, Group); };
  // Retry if fchown call fails due to interruption.
  if ((sys::RetryAfterSignal(-1, FChown)) < 0)
    return std::error_code(errno, std::generic_category());
  return std::error_code();
}

}  // end namespace fs

namespace path {

bool home_directory(SmallVectorImpl<char> &result) {
  std::unique_ptr<char[]> Buf;
  char *RequestedDir = getenv("HOME");
  if (!RequestedDir) {
    long BufSize = sysconf(_SC_GETPW_R_SIZE_MAX);
    if (BufSize <= 0) BufSize = 16384;
    Buf = std::make_unique<char[]>(BufSize);
    struct passwd Pwd;
    struct passwd *pw = nullptr;
    getpwuid_r(getuid(), &Pwd, Buf.get(), BufSize, &pw);
    if (pw && pw->pw_dir) RequestedDir = pw->pw_dir;
  }
  if (!RequestedDir) return false;

  result.clear();
  result.append(RequestedDir, RequestedDir + strlen(RequestedDir));
  return true;
}

static bool getDarwinConfDir(bool TempDir, SmallVectorImpl<char> &Result) {
  return false;
}

bool user_config_directory(SmallVectorImpl<char> &result) {
  // XDG_CONFIG_HOME as defined in the XDG Base Directory Specification:
  // http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
  if (const char *RequestedDir = getenv("XDG_CONFIG_HOME")) {
    result.clear();
    result.append(RequestedDir, RequestedDir + strlen(RequestedDir));
    return true;
  }

  // Fallback: ~/.config
  if (!home_directory(result)) {
    return false;
  }
  append(result, ".config");
  return true;
}

bool cache_directory(SmallVectorImpl<char> &result) {
  // XDG_CACHE_HOME as defined in the XDG Base Directory Specification:
  // http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
  if (const char *RequestedDir = getenv("XDG_CACHE_HOME")) {
    result.clear();
    result.append(RequestedDir, RequestedDir + strlen(RequestedDir));
    return true;
  }

  if (!home_directory(result)) {
    return false;
  }
  append(result, ".cache");
  return true;
}

static const char *getEnvTempDir() {
  // Check whether the temporary directory is specified by an environment
  // variable.
  const char *EnvironmentVariables[] = {"TMPDIR", "TMP", "TEMP", "TEMPDIR"};
  for (const char *Env : EnvironmentVariables) {
    if (const char *Dir = std::getenv(Env)) return Dir;
  }

  return nullptr;
}

static const char *getDefaultTempDir(bool ErasedOnReboot) {
#ifdef P_tmpdir
  if ((bool)P_tmpdir) return P_tmpdir;
#endif

  if (ErasedOnReboot) return "/tmp";
  return "/var/tmp";
}

void system_temp_directory(bool ErasedOnReboot, SmallVectorImpl<char> &Result) {
  Result.clear();

  if (ErasedOnReboot) {
    // There is no env variable for the cache directory.
    if (const char *RequestedDir = getEnvTempDir()) {
      Result.append(RequestedDir, RequestedDir + strlen(RequestedDir));
      return;
    }
  }

  if (getDarwinConfDir(ErasedOnReboot, Result)) return;

  const char *RequestedDir = getDefaultTempDir(ErasedOnReboot);
  Result.append(RequestedDir, RequestedDir + strlen(RequestedDir));
}

}  // namespace path

}  // namespace sys
}  // namespace domino

#endif  // DOMINO_SUPPORT_FILESYSTEM_PATH_INC_
