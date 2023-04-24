#ifndef DOMINO_UTIL_SMALLSTRING_H_
#define DOMINO_UTIL_SMALLSTRING_H_

#include <domino/util/SmallVector.h>
#include <domino/util/StringRef.h>

#include <cstddef>
#include <initializer_list>

namespace domino {

template <unsigned InternalLen>
class SmallString : public SmallVector<char, InternalLen> {
 public:
  SmallString() = default;

  SmallString(StringRef S)
      : SmallVector<char, InternalLen>(S.begin(), S.end()) {}

  SmallString(std::initializer_list<StringRef> refs)
      : SmallVector<char, InternalLen>() {
    this->append(refs);
  }

  template <typename ItTy>
  SmallString(ItTy S, ItTy E) : SmallVector<char, InternalLen>(S, E) {}

  using SmallVector<char, InternalLen>::assign;

  void assign(StringRef RHS) {
    SmallVectorImpl<char>::assign(RHS.begin(), RHS.end());
  }

  void assign(std::initializer_list<StringRef> Refs) {
    this->clear();
    append(Refs);
  }

  using SmallVector<char, InternalLen>::append;

  void append(StringRef RHS) {
    SmallVectorImpl<char>::append(RHS.begin(), RHS.end());
  }

  void append(std::initializer_list<StringRef> Refs) {
    size_t CurrentSize = this->size();
    size_t SizeNeeded = CurrentSize;
    for (const StringRef& Ref : Refs) {
      SizeNeeded += Ref.size();
    }
    this->resize_for_overwrite(SizeNeeded);
    for (const StringRef& Ref : Refs) {
      std::copy(Ref.begin(), Ref.end(), this->begin() + CurrentSize);
      CurrentSize += Ref.size();
    }
    assert(CurrentSize == this->size());
  }

  bool equals(StringRef RHS) const { return str().equals(RHS); }

  bool equals_insensitive(StringRef RHS) const {
    return str().equals_insensitive(RHS);
  }

  int compare(StringRef RHS) const { return str().compare(RHS); }

  int compare_insensitive(StringRef RHS) const {
    return str().compare_insensitive(RHS);
  }

  int compare_numeric(StringRef RHS) const {
    return str().compare_numeric(RHS);
  }

  bool starts_with(StringRef Prefix) const { return str().starts_with(Prefix); }

  bool ends_with(StringRef Suffix) const { return str().ends_with(Suffix); }

  size_t find(char C, size_t From = 0) const { return str().find(C, From); }

  size_t find(StringRef Str, size_t From = 0) const {
    return str().find(Str, From);
  }

  size_t rfind(char C, size_t From = StringRef::npos) const {
    return str().rfind(C, From);
  }

  size_t rfind(StringRef Str) const { return str().rfind(Str); }

  size_t find_first_of(char C, size_t From = 0) const {
    return str().find_first_of(C, From);
  }

  /// Complexity: O(size() + Chars.size())
  size_t find_first_of(StringRef Chars, size_t From = 0) const {
    return str().find_first_of(Chars, From);
  }

  size_t find_first_not_of(char C, size_t From = 0) const {
    return str().find_first_not_of(C, From);
  }

  /// Complexity: O(size() + Chars.size())
  size_t find_first_not_of(StringRef Chars, size_t From = 0) const {
    return str().find_first_not_of(Chars, From);
  }

  size_t find_last_of(char C, size_t From = StringRef::npos) const {
    return str().find_last_of(C, From);
  }

  /// Complexity: O(size() + Chars.size())
  size_t find_last_of(StringRef Chars, size_t From = StringRef::npos) const {
    return str().find_last_of(Chars, From);
  }

  size_t count(char C) const { return str().count(C); }

  size_t count(StringRef Str) const { return str().count(Str); }

  StringRef substr(size_t Start, size_t N = StringRef::npos) const {
    return str().substr(Start, N);
  }

  StringRef slice(size_t Start, size_t End) const {
    return str().slice(Start, End);
  }

  StringRef str() const { return StringRef(this->data(), this->size()); }

  const char* c_str() {
    this->push_back(0);
    this->pop_back();
    return this->data();
  }

  /// Implicit conversion to StringRef.
  operator StringRef() const { return str(); }

  explicit operator std::string() const {
    return std::string(this->data(), this->size());
  }

  SmallString& operator=(StringRef RHS) {
    this->assign(RHS);
    return *this;
  }

  SmallString& operator+=(StringRef RHS) {
    this->append(RHS.begin(), RHS.end());
    return *this;
  }
  SmallString& operator+=(char C) {
    this->push_back(C);
    return *this;
  }
};

}  // namespace domino

#endif  // DOMINO_UTIL_SMALLSTRING_H_