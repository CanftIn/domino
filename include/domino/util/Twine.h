#ifndef DOMINO_UTIL_TWINE_H_
#define DOMINO_UTIL_TWINE_H_

#include <domino/util/Logging.h>
#include <domino/util/SmallVector.h>
#include <domino/util/StringRef.h>

#include <cassert>
#include <cstdint>
#include <string>
#include <string_view>

namespace domino {

class formatv_object_base;
class raw_ostream;

/// Twine - A lightweight data structure for efficiently representing the
/// concatenation of temporary values as strings.
///
/// A Twine is a kind of rope, it represents a concatenated string using a
/// binary-tree, where the string is the preorder of the nodes. Since the
/// Twine can be efficiently rendered into a buffer when its result is used,
/// it avoids the cost of generating temporary values for intermediate string
/// results -- particularly in cases when the Twine result is never
/// required. By explicitly tracking the type of leaf nodes, we can also avoid
/// the creation of temporary strings for conversions operations (such as
/// appending an integer to a string).
///
/// A Twine is not intended for use directly and should not be stored, its
/// implementation relies on the ability to store pointers to temporary stack
/// objects which may be deallocated at the end of a statement. Twines should
/// only be used accepted as const references in arguments, when an API wishes
/// to accept possibly-concatenated strings.
///
/// Twines support a special 'null' value, which always concatenates to form
/// itself, and renders as an empty string. This can be returned from APIs to
/// effectively nullify any concatenations performed on the result.
///
/// \b Implementation
///
/// Given the nature of a Twine, it is not possible for the Twine's
/// concatenation method to construct interior nodes; the result must be
/// represented inside the returned value. For this reason a Twine object
/// actually holds two values, the left- and right-hand sides of a
/// concatenation. We also have nullary Twine objects, which are effectively
/// sentinel values that represent empty strings.
///
/// Thus, a Twine can effectively have zero, one, or two children. The \see
/// isNullary(), \see isUnary(), and \see isBinary() predicates exist for
/// testing the number of children.
///
/// We maintain a number of invariants on Twine objects (FIXME: Why):
///  - Nullary twines are always represented with their Kind on the left-hand
///    side, and the Empty kind on the right-hand side.
///  - Unary twines are always represented with the value on the left-hand
///    side, and the Empty kind on the right-hand side.
///  - If a Twine has another Twine as a child, that child should always be
///    binary (otherwise it could have been folded into the parent).
///
/// These invariants are check by \see isValid().
///
/// \b Efficiency Considerations
///
/// The Twine is designed to yield efficient and small code for common
/// situations. For this reason, the concat() method is inlined so that
/// concatenations of leaf nodes can be optimized into stores directly into a
/// single stack allocated object.
///
/// In practice, not all compilers can be trusted to optimize concat() fully,
/// so we provide two additional methods (and accompanying operator+
/// overloads) to guarantee that particularly important cases (cstring plus
/// StringRef) codegen as desired.
class Twine {
  /// NodeKind - Represent the type of an argument.
  enum NodeKind : unsigned char {
    /// An empty string; the result of concatenating anything with it is also
    /// empty.
    NullKind,

    /// The empty string.
    EmptyKind,

    /// A pointer to a Twine instance.
    TwineKind,

    /// A pointer to a C string instance.
    CStringKind,

    /// A pointer to an std::string instance.
    StdStringKind,

    /// A Pointer and Length representation. Used for std::string_view,
    /// StringRef, and SmallString.  Can't use a StringRef here
    /// because they are not trivally constructible.
    PtrAndLengthKind,

    /// A pointer to a formatv_object_base instance.
    FormatvObjectKind,

    /// A char value, to render as a character.
    CharKind,

    /// An unsigned int value, to render as an unsigned decimal integer.
    DecUIKind,

    /// An int value, to render as a signed decimal integer.
    DecIKind,

    /// A pointer to an unsigned long value, to render as an unsigned decimal
    /// integer.
    DecULKind,

    /// A pointer to a long value, to render as a signed decimal integer.
    DecLKind,

    /// A pointer to an unsigned long long value, to render as an unsigned
    /// decimal integer.
    DecULLKind,

    /// A pointer to a long long value, to render as a signed decimal integer.
    DecLLKind,

    /// A pointer to a uint64_t value, to render as an unsigned hexadecimal
    /// integer.
    UHexKind
  };

  union Child {
    const Twine *twine;
    const char *cString;
    const std::string *stdString;
    struct {
      const char *ptr;
      size_t length;
    } ptrAndLength;
    const formatv_object_base *formatvObject;
    char character;
    unsigned int decUI;
    int decI;
    const unsigned long *decUL;
    const long *decL;
    const unsigned long long *decULL;
    const long long *decLL;
    const uint64_t *uHex;
  };

  /// LHS - The prefix in the concatenation, which may be uninitialized for
  /// Null or Empty kinds.
  Child LHS;

  /// RHS - The suffix in the concatenation, which may be uninitialized for
  /// Null or Empty kinds.
  Child RHS;

  /// LHSKind - The NodeKind of the left hand side, \see getLHSKind().
  NodeKind LHSKind = EmptyKind;

  /// RHSKind - The NodeKind of the right hand side, \see getRHSKind().
  NodeKind RHSKind = EmptyKind;

  explicit Twine(NodeKind Kind) : LHSKind(Kind) {
    assert(isNullary() && "Invalid kind!");
  }

  explicit Twine(const Twine &LHS, const Twine &RHS)
      : LHSKind(TwineKind), RHSKind(TwineKind) {
    this->LHS.twine = &LHS;
    this->RHS.twine = &RHS;
    assert(isValid() && "Invalid twine!");
  }

  explicit Twine(Child LHS, NodeKind LHSKind, Child RHS, NodeKind RHSKind)
      : LHS(LHS), RHS(RHS), LHSKind(LHSKind), RHSKind(RHSKind) {
    assert(isValid() && "Invalid twine!");
  }

  bool isNull() const { return getLHSKind() == NullKind; }

  bool isEmpty() const { return getLHSKind() == EmptyKind; }

  bool isNullary() const { return isNull() || isEmpty(); }

  bool isUnary() const { return getRHSKind() == EmptyKind && !isNullary(); }

  bool isBinary() const {
    return getLHSKind() != NullKind && getRHSKind() != EmptyKind;
  }

  bool isValid() const {
    // Nullary twines always have Empty on the RHS.
    if (isNullary() && getRHSKind() != EmptyKind) return false;

    // Null should never appear on the RHS.
    if (getRHSKind() == NullKind) return false;

    // The RHS cannot be non-empty if the LHS is empty.
    if (getRHSKind() != EmptyKind && getLHSKind() == EmptyKind) return false;

    // A twine child should always be binary.
    if (getLHSKind() == TwineKind && !LHS.twine->isBinary()) return false;
    if (getRHSKind() == TwineKind && !RHS.twine->isBinary()) return false;

    return true;
  }

  NodeKind getLHSKind() const { return LHSKind; }

  NodeKind getRHSKind() const { return RHSKind; }

  void printOneChild(raw_ostream &OS, Child Ptr, NodeKind Kind) const;

  void printOneChildRepr(raw_ostream &OS, Child Ptr, NodeKind Kind) const;

 public:
  Twine() { assert(isValid() && "Invalid twine!"); }

  Twine(const Twine &) = default;

  Twine(const char *Str) {
    if (Str[0] != '\0') {
      LHS.cString = Str;
      LHSKind = CStringKind;
    } else
      LHSKind = EmptyKind;

    assert(isValid() && "Invalid twine!");
  }

  Twine(std::nullptr_t) = delete;

  Twine(const std::string &Str) : LHSKind(StdStringKind) {
    LHS.stdString = &Str;
    assert(isValid() && "Invalid twine!");
  }

  Twine(const std::string_view &Str) : LHSKind(PtrAndLengthKind) {
    LHS.ptrAndLength.ptr = Str.data();
    LHS.ptrAndLength.length = Str.length();
    assert(isValid() && "Invalid twine!");
  }

  Twine(const StringRef &Str) : LHSKind(PtrAndLengthKind) {
    LHS.ptrAndLength.ptr = Str.data();
    LHS.ptrAndLength.length = Str.size();
    assert(isValid() && "Invalid twine!");
  }

  Twine(const SmallVectorImpl<char> &Str) : LHSKind(PtrAndLengthKind) {
    LHS.ptrAndLength.ptr = Str.data();
    LHS.ptrAndLength.length = Str.size();
    assert(isValid() && "Invalid twine!");
  }

  Twine(const formatv_object_base &Fmt) : LHSKind(FormatvObjectKind) {
    LHS.formatvObject = &Fmt;
    assert(isValid() && "Invalid twine!");
  }

  explicit Twine(char Val) : LHSKind(CharKind) { LHS.character = Val; }

  explicit Twine(signed char Val) : LHSKind(CharKind) {
    LHS.character = static_cast<char>(Val);
  }

  explicit Twine(unsigned char Val) : LHSKind(CharKind) {
    LHS.character = static_cast<char>(Val);
  }

  explicit Twine(unsigned Val) : LHSKind(DecUIKind) { LHS.decUI = Val; }

  explicit Twine(int Val) : LHSKind(DecIKind) { LHS.decI = Val; }

  explicit Twine(const unsigned long &Val) : LHSKind(DecULKind) {
    LHS.decUL = &Val;
  }

  explicit Twine(const long &Val) : LHSKind(DecLKind) { LHS.decL = &Val; }

  explicit Twine(const unsigned long long &Val) : LHSKind(DecULLKind) {
    LHS.decULL = &Val;
  }

  explicit Twine(const long long &Val) : LHSKind(DecLLKind) {
    LHS.decLL = &Val;
  }

  Twine(const char *LHS, const StringRef &RHS)
      : LHSKind(CStringKind), RHSKind(PtrAndLengthKind) {
    this->LHS.cString = LHS;
    this->RHS.ptrAndLength.ptr = RHS.data();
    this->RHS.ptrAndLength.length = RHS.size();
    assert(isValid() && "Invalid twine!");
  }

  Twine(const StringRef &LHS, const char *RHS)
      : LHSKind(PtrAndLengthKind), RHSKind(CStringKind) {
    this->LHS.ptrAndLength.ptr = LHS.data();
    this->LHS.ptrAndLength.length = LHS.size();
    this->RHS.cString = RHS;
    assert(isValid() && "Invalid twine!");
  }

  Twine &operator=(const Twine &) = delete;

  static Twine createNull() { return Twine(NullKind); }

  static Twine utohexstr(const uint64_t &Val) {
    Child LHS, RHS;
    LHS.uHex = &Val;
    RHS.twine = nullptr;
    return Twine(LHS, UHexKind, RHS, EmptyKind);
  }

  bool isTriviallyEmpty() const { return isNullary(); }

  bool isSingleStringRef() const {
    if (getRHSKind() != EmptyKind) return false;

    switch (getLHSKind()) {
      case EmptyKind:
      case CStringKind:
      case StdStringKind:
      case PtrAndLengthKind:
        return true;
      default:
        return false;
    }
  }

  Twine concat(const Twine &Suffix) const;

  std::string str() const;

  void toVector(SmallVectorImpl<char> &Out) const;

  StringRef getSingleStringRef() const {
    assert(isSingleStringRef() && "This cannot be had as a single stringref!");
    switch (getLHSKind()) {
      default:
        DOMINO_ERROR_ABORT("Out of sync with isSingleStringRef");
      case EmptyKind:
        return StringRef();
      case CStringKind:
        return StringRef(LHS.cString);
      case StdStringKind:
        return StringRef(*LHS.stdString);
      case PtrAndLengthKind:
        return StringRef(LHS.ptrAndLength.ptr, LHS.ptrAndLength.length);
    }
  }

  StringRef toStringRef(SmallVectorImpl<char> &Out) const {
    if (isSingleStringRef()) return getSingleStringRef();
    toVector(Out);
    return StringRef(Out.data(), Out.size());
  }

  /// This returns the twine as a single null terminated StringRef if it
  /// can be represented as such. Otherwise the twine is written into the
  /// given SmallVector and a StringRef to the SmallVector's data is returned.
  ///
  /// The returned StringRef's size does not include the null terminator.
  StringRef toNullTerminatedStringRef(SmallVectorImpl<char> &Out) const;

  void print(raw_ostream &OS) const;

  void dump() const;

  void printRepr(raw_ostream &OS) const;

  /// Dump the representation of this twine to stderr.
  void dumpRepr() const;
};

inline Twine Twine::concat(const Twine &Suffix) const {
  // Concatenation with null is null.
  if (isNull() || Suffix.isNull()) return Twine(NullKind);

  // Concatenation with empty yields the other side.
  if (isEmpty()) return Suffix;
  if (Suffix.isEmpty()) return *this;

  // Otherwise we need to create a new node, taking care to fold in unary
  // twines.
  Child NewLHS, NewRHS;
  NewLHS.twine = this;
  NewRHS.twine = &Suffix;
  NodeKind NewLHSKind = TwineKind, NewRHSKind = TwineKind;
  if (isUnary()) {
    NewLHS = LHS;
    NewLHSKind = getLHSKind();
  }
  if (Suffix.isUnary()) {
    NewRHS = Suffix.LHS;
    NewRHSKind = Suffix.getLHSKind();
  }

  return Twine(NewLHS, NewLHSKind, NewRHS, NewRHSKind);
}

inline Twine operator+(const Twine &LHS, const Twine &RHS) {
  return LHS.concat(RHS);
}

inline Twine operator+(const char *LHS, const StringRef &RHS) {
  return Twine(LHS, RHS);
}

inline Twine operator+(const StringRef &LHS, const char *RHS) {
  return Twine(LHS, RHS);
}

inline raw_ostream &operator<<(raw_ostream &OS, const Twine &RHS) {
  RHS.print(OS);
  return OS;
}

}  // namespace domino

#endif  // DOMINO_UTIL_TWINE_H_