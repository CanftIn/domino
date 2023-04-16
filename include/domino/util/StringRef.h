#ifndef DOMINO_UTIL_STRINGREF_H_
#define DOMINO_UTIL_STRINGREF_H_

#include <domino/util/IteratorRange.h>

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

  [[nodiscard]] bool equals_insenstive(StringRef RHS) const {
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

  [[nodiscard]] size_t find(char C, size_t From = 0) const {
    return std::string_view(*this).find(C, From);
  }

  [[nodiscard]] size_t find_insensitive(char C, size_t From = 0) const;

  //[[nodiscard]] size_t find_if(function_ref<bool(char)> F,
  //                             size_t From = 0) const {
  //  StringRef S = drop_front(From);
  //}

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

  [[nodiscard]] StringRef drop_front(size_t N = 1) const {
    assert(size() >= N && "Dropping more elements than exist");
    return substr(N);
  }

  [[nodiscard]] StringRef drop_back(size_t N = 1) const {
    assert(size() >= N && "Dropping more elements than exist");
    return substr(0, size() - N);
  }

  bool consume_front(StringRef Prefix) {
    if (!starts_with(Prefix)) return false;
  }
};

}  // namespace domino

#endif  // DOMINO_UTIL_STRINGREF_H_