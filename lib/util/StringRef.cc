#include <domino/util/Macros.h>
#include <domino/util/StringExtras.h>
#include <domino/util/StringRef.h>

#include <bitset>
#include <limits.h>

using namespace domino;

constexpr size_t StringRef::npos;

static int ascii_strncasecmp(const char* LHS, const char* RHS, size_t Length) {
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
         ascii_strncasecmp(end() - Suffix.Length, Suffix.Data, Suffix.Length) ==
             0;
}

size_t StringRef::find_insensitive(char C, size_t From) const {
  char L = toLower(C);
  return find_if([L](char D) { return toLower(D) == L; }, From);
}

size_t StringRef::find(StringRef Str, size_t From) const {
  if (From > Length) return npos;

  const char* Start = Data + From;
  size_t Size = Length - From;

  const char* Needle = Str.data();
  size_t N = Str.size();
  if (N == 0) return From;
  if (Size < N) return npos;
  if (N == 1) {
    const char* Ptr = (const char*)::memchr(Start, Needle[0], Size);
    return Ptr == nullptr ? npos : Ptr - Data;
  }

  const char* Stop = Start + (Size - N + 1);
  if (N == 2) {
    // Provide a fast path for newline finding (CRLF case) in InclusionRewriter.
    // Not the most optimized strategy, but getting memcmp inlined should be
    // good enough.
    do {
      if (std::memcmp(Start, Needle, 2) == 0) return Start - Data;
      ++Start;
    } while (Start < Stop);
    return npos;
  }

  // For short haystacks or unsupported needles fall back to the naive algorithm
  if (Size < 16 || N > 255) {
    do {
      if (std::memcmp(Start, Needle, N) == 0) return Start - Data;
      ++Start;
    } while (Start < Stop);
    return npos;
  }

  // Build the bad char heuristic table, with uint8_t to reduce cache thrashing.
  uint8_t BadCharSkip[256];
  std::memset(BadCharSkip, N, 256);
  for (unsigned i = 0; i != N - 1; ++i)
    BadCharSkip[(uint8_t)Str[i]] = N - 1 - i;

  do {
    uint8_t Last = Start[N - 1];
    if (DOMINO_UNLIKELY(Last == (uint8_t)Needle[N - 1]))
      if (std::memcmp(Start, Needle, N - 1) == 0) return Start - Data;

    // Otherwise skip the appropriate number of bytes.
    Start += BadCharSkip[Last];
  } while (Start < Stop);

  return npos;
}

size_t StringRef::find_insensitive(StringRef Str, size_t From) const {
  StringRef This = substr(From);
  while (This.size() >= Str.size()) {
    if (This.starts_with_insensitive(Str)) return From;
    This = This.drop_front();
    ++From;
  }
  return npos;
}

size_t StringRef::rfind_insensitive(char C, size_t From) const {
  From = std::min(From, Length);
  size_t i = From;
  while (i != 0) {
    --i;
    if (toLower(Data[i]) == toLower(C)) return i;
  }
  return npos;
}

size_t StringRef::rfind_insensitive(StringRef Str) const {
  size_t N = Str.size();
  if (N > Length) return npos;
  for (size_t i = Length - N + 1, e = 0; i != e;) {
    --i;
    if (substr(i, N).equals_insensitive(Str)) return i;
  }
  return npos;
}

StringRef::size_type StringRef::find_first_of(StringRef Chars,
                                              size_t From) const {
  std::bitset<1 << CHAR_BIT> CharBits;
  for (char C : Chars) CharBits.set((unsigned char)C);

  for (size_type i = std::min(From, Length), e = Length; i != e; ++i)
    if (CharBits.test((unsigned char)Data[i])) return i;
  return npos;
}


/// find_first_not_of - Find the first character in the string that is not
/// \arg C or npos if not found.
StringRef::size_type StringRef::find_first_not_of(char C, size_t From) const {
  return std::string_view(*this).find_first_not_of(C, From);
}

/// find_first_not_of - Find the first character in the string that is not
/// in the string \arg Chars, or npos if not found.
///
/// Note: O(size() + Chars.size())
StringRef::size_type StringRef::find_first_not_of(StringRef Chars,
                                                  size_t From) const {
  std::bitset<1 << CHAR_BIT> CharBits;
  for (char C : Chars)
    CharBits.set((unsigned char)C);

  for (size_type i = std::min(From, Length), e = Length; i != e; ++i)
    if (!CharBits.test((unsigned char)Data[i]))
      return i;
  return npos;
}

/// find_last_of - Find the last character in the string that is in \arg C,
/// or npos if not found.
///
/// Note: O(size() + Chars.size())
StringRef::size_type StringRef::find_last_of(StringRef Chars,
                                             size_t From) const {
  std::bitset<1 << CHAR_BIT> CharBits;
  for (char C : Chars)
    CharBits.set((unsigned char)C);

  for (size_type i = std::min(From, Length) - 1, e = -1; i != e; --i)
    if (CharBits.test((unsigned char)Data[i]))
      return i;
  return npos;
}

/// find_last_not_of - Find the last character in the string that is not
/// \arg C, or npos if not found.
StringRef::size_type StringRef::find_last_not_of(char C, size_t From) const {
  for (size_type i = std::min(From, Length) - 1, e = -1; i != e; --i)
    if (Data[i] != C)
      return i;
  return npos;
}

/// find_last_not_of - Find the last character in the string that is not in
/// \arg Chars, or npos if not found.
///
/// Note: O(size() + Chars.size())
StringRef::size_type StringRef::find_last_not_of(StringRef Chars,
                                                 size_t From) const {
  std::bitset<1 << CHAR_BIT> CharBits;
  for (char C : Chars)
    CharBits.set((unsigned char)C);

  for (size_type i = std::min(From, Length) - 1, e = -1; i != e; --i)
    if (!CharBits.test((unsigned char)Data[i]))
      return i;
  return npos;
}