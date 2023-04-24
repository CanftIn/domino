#ifndef DOMINO_SUPPORT_FORMATVARIADICDETAILS_H_
#define DOMINO_SUPPORT_FORMATVARIADICDETAILS_H_

#include <domino/support/raw_ostream.h>
#include <domino/util/STLExtras.h>
#include <domino/util/StringRef.h>

#include <type_traits>

namespace domino {

template <typename T, typename Enable = void>
struct format_provider;

class Error;

namespace detail {

class format_adapter {
  virtual void achor();

 protected:
  virtual ~format_adapter() = default;

 public:
  virtual void format(raw_ostream& OS, StringRef Options) = 0;
};

template <typename T>
class provider_format_adapter : public format_adapter {
  T Item;

 public:
  explicit provider_format_adapter(T&& Item) : Item(std::forward<T>(Item)) {}

  void format(domino::raw_ostream& S, StringRef Options) override {
    format_provider<std::decay_t<T>>::format(Item, S, Options);
  }
};

template <typename T>
class stream_operator_format_adapter : public format_adapter {
  T Item;

 public:
  explicit stream_operator_format_adapter(T&& Item)
      : Item(std::forward<T>(Item)) {}

  void format(domino::raw_ostream& S, StringRef Options) override { S << Item; }
};

template <typename T>
class missing_format_adapter;

template <typename T>
class has_FormatProvider {
 public:
  using Decayed = std::decay_t<T>;
  using Signature_format = void (*)(const Decayed, domino::raw_ostream&,
                                    StringRef);

  template <typename U>
  static char test(SameType<Signature_format, &U::format>*);

  template <typename U>
  static double test(...);

  static bool const value =
      (sizeof(test<domino::format_provider<Decayed>>(nullptr)) == 1);
};

template <typename T>
class has_StreamOperator {
 public:
  using ConstRefT = const std::decay_t<T>&;

  template <typename U>
  static char test(std::enable_if_t<
                   std::is_same<decltype(std::declval<domino::raw_ostream&>()
                                         << std::declval<U>()),
                                domino::raw_ostream&>::value,
                   int*>);

  template <typename U>
  static double test(...);

  static bool const value = (sizeof(test<ConstRefT>(nullptr)) == 1);
};

template <typename T>
struct uses_format_member
    : public std::integral_constant<
          bool,
          std::is_base_of<format_adapter, std::remove_reference_t<T>>::value> {
};

template <typename T>
struct uses_format_provider
    : public std::integral_constant<bool, !uses_format_member<T>::value &&
                                              has_FormatProvider<T>::value> {};

template <typename T>
struct uses_stream_operator
    : public std::integral_constant<bool, !uses_format_member<T>::value &&
                                              !uses_format_provider<T>::value &&
                                              has_StreamOperator<T>::value> {};

template <typename T>
struct uses_missing_provider
    : public std::integral_constant<bool, !uses_format_member<T>::value &&
                                              !uses_format_provider<T>::value &&
                                              !uses_stream_operator<T>::value> {
};

template <typename T>
std::enable_if_t<uses_format_member<T>::value, T> build_format_adapter(
    T&& Item) {
  return std::forward<T>(Item);
}

template <typename T>
std::enable_if_t<uses_format_provider<T>::value, provider_format_adapter<T>>
build_format_adapter(T&& Item) {
  return provider_format_adapter<T>(std::forward<T>(Item));
}

template <typename T>
std::enable_if_t<uses_stream_operator<T>::value,
                 stream_operator_format_adapter<T>>
build_format_adapter(T&& Item) {
  static_assert(!std::is_same<domino::Error, std::remove_cv_t<T>>::value,
                "Error-by-value must be wrapped in fmt_consume() for formatv");
  return stream_operator_format_adapter<T>(std::forward<T>(Item));
}

template <typename T>
std::enable_if_t<uses_missing_provider<T>::value, missing_format_adapter<T>>
build_format_adapter(T&&) {
  return missing_format_adapter<T>();
}

}  // namespace detail
}  // namespace domino

#endif  // DOMINO_SUPPORT_FORMATVARIADICDETAILS_H_