#ifndef DOMINO_UTIL_STRINGREF_H_
#define DOMINO_UTIL_STRINGREF_H_

#include <domino/util/IteratorRange.h>
#include <domino/util/STLExtras.h>
#include <domino/util/SmallVector.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>

namespace domino {

class StringRef {
 public:
  static constexpr size_t npos = ~size_t(0);

  using iterator = const char*;
  using const_iterator = const char*;
  using size_type = size_t;

 private:
  const char* Data = nullptr;

  size_t Length = 0;

  static int compareMemory(const char* LHS, const char* RHS, size_t Length) {
    if (Length == 0) return 0;
    return ::memcmp(LHS, RHS, Length);
  }

 public:
  StringRef() = default;

  StringRef(std::nullptr_t) = delete;

  constexpr StringRef(const char* Str)
      : Data(Str), Length(Str ? std::char_traits<char>::length(Str) : 0) {}

  constexpr StringRef(const char* data, size_t length)
      : Data(data), Length(length) {}

  StringRef(const std::string& Str) : Data(Str.data()), Length(Str.length()) {}

  constexpr StringRef(std::string_view Str)
      : Data(Str.data()), Length(Str.size()) {}

  iterator begin() const { return Data; }

  iterator end() const { return Data + Length; }

  const unsigned char* bytes_begin() const {
    return reinterpret_cast<const unsigned char*>(begin());
  }

  const unsigned char* bytes_end() const {
    return reinterpret_cast<const unsigned char*>(end());
  }

  iterator_range<const unsigned char*> bytes() const {
    return make_range(bytes_begin(), bytes_end());
  }

  [[nodiscard]] const char* data() const { return Data; }

  [[nodiscard]] constexpr bool empty() const { return Length == 0; }

  [[nodiscard]] constexpr size_t size() const { return Length; }

  [[nodiscard]] char front() const {
    assert(!empty());
    return Data[0];
  }

  [[nodiscard]] char back() const {
    assert(!empty());
    return Data[Length - 1];
  }

  template <typename Allocator>
  [[nodiscard]] StringRef copy(Allocator& A) const {
    if (empty()) return StringRef();
    char* S = A.template Allocate<char>(Length);
    std::copy(begin(), end(), S);
    return StringRef(S, Length);
  }

  [[nodiscard]] bool equals(StringRef RHS) const {
    return (Length == RHS.Length && compareMemory(Data, RHS.Data, Length) == 0);
  }

  [[nodiscard]] bool equals_insensitive(StringRef RHS) const {
    return Length == RHS.Length && compare_insensitive(RHS) == 0;
  }

  [[nodiscard]] int compare(StringRef RHS) const {
    if (int Res = compareMemory(Data, RHS.Data, std::min(Length, RHS.Length)))
      return Res < 0 ? -1 : 1;

    if (Length == RHS.Length) return 0;
    return Length < RHS.Length ? -1 : 1;
  }

  [[nodiscard]] int compare_insensitive(StringRef RHS) const;

  [[nodiscard]] int compare_numeric(StringRef RHS) const;

  /// Determine the edit distance between this string and another
  /// string.
  ///
  /// \param Other the string to compare this string against.
  ///
  /// \param AllowReplacements whether to allow character
  /// replacements (change one character into another) as a single
  /// operation, rather than as two operations (an insertion and a
  /// removal).
  ///
  /// \param MaxEditDistance If non-zero, the maximum edit distance that
  /// this routine is allowed to compute. If the edit distance will exceed
  /// that maximum, returns \c MaxEditDistance+1.
  ///
  /// \returns the minimum number of character insertions, removals,
  /// or (if \p AllowReplacements is \c true) replacements needed to
  /// transform one of the given strings into the other. If zero,
  /// the strings are identical.
  [[nodiscard]] unsigned edit_distance(StringRef Other,
                                       bool AllowReplacements = true,
                                       unsigned MaxEditDistance = 0) const;

  [[nodiscard]] unsigned edit_distance_insensitive(
      StringRef Other, bool AllowReplacements = true,
      unsigned MaxEditDistance = 0) const;

  /// str - Get the contents as an std::string.
  [[nodiscard]] std::string str() const {
    if (!Data) return std::string();
    return std::string(Data, Length);
  }

  [[nodiscard]] char operator[](size_t Index) const {
    assert(Index < Length && "Invalid index!");
    return Data[Index];
  }

  template <typename T>
  std::enable_if_t<std::is_same<T, std::string>::value, StringRef>& operator=(
      T&& RHS) = delete;

  operator std::string_view() const { return std::string_view(data(), size()); }

  [[nodiscard]] bool starts_with(StringRef Prefix) const {
    return Length >= Prefix.Length &&
           compareMemory(Data, Prefix.Data, Prefix.Length) == 0;
  }

  /// ignoring case
  [[nodiscard]] bool starts_with_insensitive(StringRef Prefix) const;

  [[nodiscard]] bool ends_with(StringRef Suffix) const {
    return Length >= Suffix.Length &&
           compareMemory(end() - Suffix.Length, Suffix.Data, Suffix.Length) ==
               0;
  }

  [[nodiscard]] bool ends_with_insensitive(StringRef Suffix) const;

  /// Search for the first character \p C in the string.
  ///
  /// \returns The index of the first occurrence of \p C, or npos if not
  /// found.
  [[nodiscard]] size_t find(char C, size_t From = 0) const {
    return std::string_view(*this).find(C, From);
  }

  /// Search for the first character \p C in the string, ignoring case.
  ///
  /// \returns The index of the first occurrence of \p C, or npos if not
  /// found.
  [[nodiscard]] size_t find_insensitive(char C, size_t From = 0) const;

  /// Search for the first character satisfying the predicate \p F
  ///
  /// \returns The index of the first character satisfying \p F starting from
  /// \p From, or npos if not found.
  [[nodiscard]] size_t find_if(function_ref<bool(char)> F,
                               size_t From = 0) const {
    StringRef S = drop_front(From);
    while (!S.empty()) {
      if (F(S.front())) return size() - S.size();
      S = S.drop_front();
    }
    return npos;
  }

  [[nodiscard]] size_t find_if_not(function_ref<bool(char)> F,
                                   size_t From = 0) const {
    return find_if([F](char c) { return !F(c); }, From);
  }

  [[nodiscard]] size_t find(StringRef Str, size_t From = 0) const;

  [[nodiscard]] size_t find_insensitive(StringRef Str, size_t From = 0) const;

  [[nodiscard]] size_t rfind(char C, size_t From = npos) const {
    From = std::min(From, Length);
    size_t i = From;
    while (i != 0) {
      --i;
      if (Data[i] == C) return i;
    }
    return npos;
  }

  [[nodiscard]] size_t rfind(StringRef Str) const {
    return std::string_view(*this).rfind(Str);
  }

  [[nodiscard]] size_t rfind_insensitive(char C, size_t From = npos) const;

  [[nodiscard]] size_t rfind_insensitive(StringRef Str) const;

  [[nodiscard]] size_t find_first_of(char C, size_t From = 0) const {
    return find(C, From);
  }

  [[nodiscard]] size_t find_first_of(StringRef Chars, size_t From = 0) const;

  [[nodiscard]] size_t find_first_not_of(char C, size_t From = 0) const;

  [[nodiscard]] size_t find_first_not_of(StringRef Chars,
                                         size_t From = 0) const;

  [[nodiscard]] size_t find_last_of(char C, size_t From = npos) const {
    return rfind(C, From);
  }

  [[nodiscard]] size_t find_last_of(StringRef Chars, size_t From = npos) const;

  [[nodiscard]] size_t find_last_not_of(char C, size_t From = npos) const;

  [[nodiscard]] size_t find_last_not_of(StringRef Chars,
                                        size_t From = npos) const;

  [[nodiscard]] bool contains(StringRef Other) const {
    return find(Other) != npos;
  }

  [[nodiscard]] bool contains(char C) const { return find_first_of(C) != npos; }

  [[nodiscard]] bool contains_insensitive(StringRef Other) const {
    return find_insensitive(Other) != npos;
  }

  [[nodiscard]] bool contains_insensitive(char C) const {
    return find_insensitive(C) != npos;
  }

  [[nodiscard]] size_t count(char C) const {
    size_t Count = 0;
    for (size_t i = 0, e = Length; i != e; ++i)
      if (Data[i] == C) ++Count;
    return Count;
  }

  size_t count(StringRef Str) const;

  [[nodiscard]] std::string lower() const;

  [[nodiscard]] std::string upper() const;

  [[nodiscard]] constexpr StringRef substr(size_t Start,
                                           size_t N = npos) const {
    Start = std::min(Start, Length);
    return StringRef(Data + Start, std::min(N, Length - Start));
  }

  [[nodiscard]] StringRef take_front(size_t N = 1) const {
    if (N >= size()) return *this;
    return drop_back(size() - N);
  }

  [[nodiscard]] StringRef take_back(size_t N = 1) const {
    if (N >= size()) return *this;
    return drop_front(size() - N);
  }

  [[nodiscard]] StringRef take_while(function_ref<bool(char)> F) const {
    return substr(0, find_if_not(F));
  }

  [[nodiscard]] StringRef take_until(function_ref<bool(char)> F) const {
    return substr(0, find_if(F));
  }

  [[nodiscard]] StringRef drop_front(size_t N = 1) const {
    assert(size() >= N && "Dropping more elements than exist");
    return substr(N);
  }

  [[nodiscard]] StringRef drop_back(size_t N = 1) const {
    assert(size() >= N && "Dropping more elements than exist");
    return substr(0, size() - N);
  }

  [[nodiscard]] StringRef drop_while(function_ref<bool(char)> F) const {
    return substr(find_if_not(F));
  }

  [[nodiscard]] StringRef drop_until(function_ref<bool(char)> F) const {
    return substr(find_if(F));
  }

  bool consume_front(StringRef Prefix) {
    if (!starts_with(Prefix)) return false;

    *this = drop_front(Prefix.size());
    return true;
  }

  bool consume_front_insensitive(StringRef Prefix) {
    if (!starts_with_insensitive(Prefix)) return false;

    *this = drop_front(Prefix.size());
    return true;
  }

  bool consume_back(StringRef Suffix) {
    if (!ends_with(Suffix)) return false;

    *this = drop_back(Suffix.size());
    return true;
  }

  bool consume_back_insensitive(StringRef Suffix) {
    if (!ends_with_insensitive(Suffix)) return false;

    *this = drop_back(Suffix.size());
    return true;
  }

  [[nodiscard]] StringRef slice(size_t Start, size_t End) const {
    Start = std::min(Start, Length);
    End = std::min(std::max(Start, End), Length);
    return StringRef(Data + Start, End - Start);
  }

  [[nodiscard]] std::pair<StringRef, StringRef> split(char Separator) const {
    return split(StringRef(&Separator, 1));
  }

  [[nodiscard]] std::pair<StringRef, StringRef> split(
      StringRef Separator) const {
    size_t Idx = find(Separator);
    if (Idx == npos) return std::make_pair(*this, StringRef());
    return std::make_pair(slice(0, Idx), slice(Idx + Separator.size(), npos));
  }

  [[nodiscard]] std::pair<StringRef, StringRef> rsplit(
      StringRef Separator) const {
    size_t Idx = rfind(Separator);
    if (Idx == npos) return std::make_pair(*this, StringRef());
    return std::make_pair(slice(0, Idx), slice(Idx + Separator.size(), npos));
  }

  void split(SmallVectorImpl<StringRef>& A, StringRef Separator,
             int MaxSplit = -1, bool KeepEmpty = true) const;

  void split(SmallVectorImpl<StringRef>& A, char Separator, int MaxSplit = -1,
             bool KeepEmpty = true) const;

  [[nodiscard]] std::pair<StringRef, StringRef> rsplit(char Separator) const {
    return rsplit(StringRef(&Separator, 1));
  }

  [[nodiscard]] StringRef ltrim(char Char) const {
    return drop_front(std::min(Length, find_first_not_of(Char)));
  }

  [[nodiscard]] StringRef ltrim(StringRef Chars = " \t\n\v\f\r") const {
    return drop_front(std::min(Length, find_first_not_of(Chars)));
  }

  [[nodiscard]] StringRef rtrim(char Char) const {
    return drop_back(Length - std::min(Length, find_last_not_of(Char) + 1));
  }

  [[nodiscard]] StringRef rtrim(StringRef Chars = " \t\n\v\f\r") const {
    return drop_back(Length - std::min(Length, find_last_not_of(Chars) + 1));
  }

  [[nodiscard]] StringRef trim(char Char) const {
    return ltrim(Char).rtrim(Char);
  }

  [[nodiscard]] StringRef trim(StringRef Chars = " \t\n\v\f\r") const {
    return ltrim(Chars).rtrim(Chars);
  }

  [[nodiscard]] StringRef detectEOL() const {
    size_t Pos = find('\r');
    if (Pos == npos) {
      // If there is no carriage return, assume unix
      return "\n";
    }
    if (Pos + 1 < Length && Data[Pos + 1] == '\n') return "\r\n";  // Windows
    if (Pos > 0 && Data[Pos - 1] == '\n') return "\n\r";  // You monster!
    return "\r";                                          // Classic Mac
  }
};

class StringLiteral : public StringRef {
 private:
  constexpr StringLiteral(const char* Str, size_t N) : StringRef(Str, N) {}

 public:
  template <size_t N>
  constexpr StringLiteral(const char (&Str)[N]) : StringRef(Str, N - 1) {}

  // Explicit construction for strings like "foo\0bar".
  template <size_t N>
  static constexpr StringLiteral withInnerNUL(const char (&Str)[N]) {
    return StringLiteral(Str, N - 1);
  }
};

inline bool operator==(StringRef LHS, StringRef RHS) { return LHS.equals(RHS); }

inline bool operator!=(StringRef LHS, StringRef RHS) { return !(LHS == RHS); }

inline bool operator<(StringRef LHS, StringRef RHS) {
  return LHS.compare(RHS) < 0;
}

inline bool operator<=(StringRef LHS, StringRef RHS) {
  return LHS.compare(RHS) <= 0;
}

inline bool operator>(StringRef LHS, StringRef RHS) {
  return LHS.compare(RHS) > 0;
}

inline bool operator>=(StringRef LHS, StringRef RHS) {
  return LHS.compare(RHS) >= 0;
}

inline std::string& operator+=(std::string& buffer, StringRef string) {
  return buffer.append(string.data(), string.size());
}

}  // namespace domino

#endif  // DOMINO_UTIL_STRINGREF_H_