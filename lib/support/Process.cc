#include <domino/support/FileSystem.h>
#include <domino/support/Process.h>
#include <domino/support/Program.h>
#include <domino/support/filesystem/Path.h>
#include <domino/util/STLExtras.h>
#include <domino/util/StringExtras.h>
#include <stdlib.h>  // for _Exit

#include <optional>

namespace domino {
namespace sys {

std::optional<std::string> Process::FindInEnvPath(StringRef EnvName,
                                                  StringRef FileName,
                                                  char Separator) {
  return FindInEnvPath(EnvName, FileName, {}, Separator);
}

std::optional<std::string> Process::FindInEnvPath(
    StringRef EnvName, StringRef FileName, ArrayRef<std::string> IgnoreList,
    char Separator) {
  assert(!path::is_absolute(FileName));
  std::optional<std::string> FoundPath;
  std::optional<std::string> OptPath = Process::GetEnv(EnvName);
  if (!OptPath) return FoundPath;

  const char EnvPathSeparatorStr[] = {Separator, '\0'};
  SmallVector<StringRef, 8> Dirs;
  SplitString(*OptPath, Dirs, EnvPathSeparatorStr);

  for (StringRef Dir : Dirs) {
    if (Dir.empty()) continue;

    if (any_of(IgnoreList, [&](StringRef S) { return fs::equivalent(S, Dir); }))
      continue;

    SmallString<128> FilePath(Dir);
    path::append(FilePath, FileName);
    if (fs::exists(Twine(FilePath))) {
      FoundPath = std::string(FilePath.str());
      break;
    }
  }

  return FoundPath;
}

#define COLOR(FGBG, CODE, BOLD) "\033[0;" BOLD FGBG CODE "m"

#define ALLCOLORS(FGBG, BOLD)                                                  \
  {                                                                            \
    COLOR(FGBG, "0", BOLD), COLOR(FGBG, "1", BOLD), COLOR(FGBG, "2", BOLD),    \
        COLOR(FGBG, "3", BOLD), COLOR(FGBG, "4", BOLD),                        \
        COLOR(FGBG, "5", BOLD), COLOR(FGBG, "6", BOLD), COLOR(FGBG, "7", BOLD) \
  }

static const char colorcodes[2][2][8][10] = {
    {ALLCOLORS("3", ""), ALLCOLORS("3", "1;")},
    {ALLCOLORS("4", ""), ALLCOLORS("4", "1;")}};

bool Process::AreCoreFilesPrevented() { return coreFilesPrevented; }

// TODO: make robust
[[noreturn]] void Process::Exit(int RetCode, bool NoCleanup) {
  ::exit(RetCode);
}

}  // namespace sys
}  // namespace domino

#include "Process.inc"