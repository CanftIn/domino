#include <domino/util/StringExtras.h>
#include <domino/util/StringRef.h>

using namespace domino;

constexpr size_t StringRef::npos;

static int ascii_strncasecmp(const char *LHS, const char *RHS, size_t Length) {
  for (size_t I = 0; I < Length; ++I) {
    unsigned char LHC = toLower(LHS[I]);
    unsigned char RHC = toLower(RHS[I]);
    if (LHC != RHC) return LHC < RHC ? -1 : 1;
  }
  return 0;
}

int StringRef::compare_insensitive(StringRef RHS) const {
  if (int Res = ascii_strncasecmp(Data, RHS.Data, std::min(Length, RHS.Length)))
    return Res;
  if (Length == RHS.Length) return 0;
  return Length < RHS.Length ? -1 : 1;
}

bool StringRef::starts_with_insensitive(StringRef Prefix) const {
  return Length >= Prefix.Length &&
         ascii_strncasecmp(Data, Prefix.Data, Prefix.Length) == 0;
}

bool StringRef::ends_with_insensitive(StringRef Suffix) const {
  return Length >= Suffix.Length &&
         ascii_strncasecmp(end() - Suffix.Length, Suffix.Data, Suffix.Length) == 0;
}