#ifndef DOMINO_UTIL_TO_STRING_H_
#define DOMINO_UTIL_TO_STRING_H_

#include <domino/util/StringRef.h>

#include <cmath>
#include <limits>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace domino {

template <typename T>
std::string to_string(const T& t) {
  std::ostringstream o;
  o.imbue(std::locale("C"));
  o << t;
  return o.str();
}

inline std::string to_string(const float& t) {
  std::ostringstream o;
  o.imbue(std::locale("C"));
  o.precision(static_cast<std::streamsize>(std::ceil(static_cast<double>(
      std::numeric_limits<float>::digits * std::log10(2.0f) + 1))));
  o << t;
  return o.str();
}

inline std::string to_string(const double& t) {
  std::ostringstream o;
  o.imbue(std::locale("C"));
  o.precision(static_cast<std::streamsize>(std::ceil(static_cast<double>(
      std::numeric_limits<double>::digits * std::log10(2.0f) + 1))));
  o << t;
  return o.str();
}

inline std::string to_string(const long double& t) {
  std::ostringstream o;
  o.imbue(std::locale("C"));
  o.precision(static_cast<std::streamsize>(std::ceil(static_cast<double>(
      std::numeric_limits<long double>::digits * std::log10(2.0f) + 1))));
  o << t;
  return o.str();
}

template <typename K, typename V>
std::string to_string(const std::map<K, V>& m);

template <typename T>
std::string to_string(const std::set<T>& s);

template <typename T>
std::string to_string(const std::vector<T>& t);

template <typename K, typename V>
std::string to_string(const typename std::pair<K, V>& v) {
  std::ostringstream o;
  o << to_string(v.first) << ": " << to_string(v.second);
  return o.str();
}

template <typename T>
std::string to_string(const T& beg, const T& end) {
  std::ostringstream o;
  for (T it = beg; it != end; ++it) {
    if (it != beg) o << ", ";
    o << to_string(*it);
  }
  return o.str();
}

template <typename T>
std::string to_string(const std::vector<T>& t) {
  std::ostringstream o;
  o << "[" << to_string(t.begin(), t.end()) << "]";
  return o.str();
}

template <typename K, typename V>
std::string to_string(const std::map<K, V>& m) {
  std::ostringstream o;
  o << "{" << to_string(m.begin(), m.end()) << "}";
  return o.str();
}

template <typename T>
std::string to_string(const std::set<T>& s) {
  std::ostringstream o;
  o << "{" << to_string(s.begin(), s.end()) << "}";
  return o.str();
}

std::string to_string(const StringRef& s) {
  std::ostringstream o;
  for (auto it = s.begin(); it != s.end(); ++it) {
    o << to_string(*it);
  }
  return o.str();
}

}  // namespace domino

#endif  // DOMINO_UTIL_TO_STRING_H_