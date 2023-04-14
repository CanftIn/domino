#ifndef DOMINO_UTIL_JSON_H_
#define DOMINO_UTIL_JSON_H_

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace domino {

namespace detail {

template <typename T>
[[nodiscard]] constexpr auto parse_num(const std::string_view t_str) noexcept ->
    typename std::enable_if<std::is_integral<T>::value, T>::type {
  T t = 0;
  for (const auto c : t_str) {
    if (c < '0' || c > '9') {
      return t;
    }
    t *= 10;
    t += c - '0';
  }
  return t;
}

template <typename T>
[[nodiscard]] auto parse_num(const std::string_view t_str) ->
    typename std::enable_if<!std::is_integral<T>::value, T>::type {
  T t = 0;
  T base{};
  T decimal_place = 0;
  int exponent = 0;

  for (const auto c : t_str) {
    switch (c) {
      case '.':
        decimal_place = 10;
        break;
      case 'e':
      case 'E':
        exponent = 1;
        decimal_place = 0;
        base = t;
        t = 0;
        break;
      case '-':
        exponent = -1;
        break;
      case '+':
        break;
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        if (decimal_place < 10) {
          t *= 10;
          t += static_cast<T>(c - '0');
        } else {
          t += static_cast<T>(c - '0') / decimal_place;
          decimal_place *= 10;
        }
        break;
      default:
        break;
    }
  }
  return exponent ? base * std::pow(T(10), t * static_cast<T>(exponent)) : t;
}

}  // namespace detail

class Json {
 public:
  enum class Class {
    Null = 0,
    Object,
    Array,
    String,
    Floating,
    Integral,
    Boolean
  };

 private:
  using Data =
      std::variant<std::nullptr_t, std::map<std::string, Json>,
                   std::vector<Json>, std::string, double, std::int64_t, bool>;

  struct Internal {
    Internal(std::nullptr_t) : _d(nullptr) {}
    Internal() : _d(nullptr) {}
    Internal(Class c) : _d(make_type(c)) {}
    template <typename T>
    Internal(T t) : _d(std::move(t)) {}

    static Data make_type(Class c) {
      switch (c) {
        case Class::Null:
          return nullptr;
        case Class::Object:
          return std::map<std::string, Json>{};
        case Class::Array:
          return std::vector<Json>{};
        case Class::String:
          return std::string{};
        case Class::Floating:
          return double{};
        case Class::Integral:
          return std::int64_t{};
        case Class::Boolean:
          return bool{};
      }
      throw std::runtime_error("unknown type");
    }

    void set_type(Class c) {
      if (type() != c) {
        _d = make_type(c);
      }
    }

    Class type() const noexcept { return Class(_d.index()); }

    template <auto ClassValue, typename Visitor, typename Or>
    decltype(auto) visitor(Visitor &&v, Or &&o) const {
      if (type() == Class(ClassValue)) {
        return v(std::get<static_cast<std::size_t>(ClassValue)>(_d));
      } else {
        return o();
      }
    }

    template <auto ClassValue>
    auto &get_set_type() {
      set_type(ClassValue);
      return (std::get<static_cast<std::size_t>(ClassValue)>(_d));
    }

    auto &Map() { return get_set_type<Class::Object>(); }
    auto &Vector() { return get_set_type<Class::Array>(); }
    auto &String() { return get_set_type<Class::String>(); }
    auto &Int() { return get_set_type<Class::Integral>(); }
    auto &Float() { return get_set_type<Class::Floating>(); }
    auto &Bool() { return get_set_type<Class::Boolean>(); }

    auto Map() const noexcept {
      return std::get_if<static_cast<std::size_t>(Class::Object)>(&_d);
    }
    auto Vector() const noexcept {
      return std::get_if<static_cast<std::size_t>(Class::Array)>(&_d);
    }
    auto String() const noexcept {
      return std::get_if<static_cast<std::size_t>(Class::String)>(&_d);
    }
    auto Int() const noexcept {
      return std::get_if<static_cast<std::size_t>(Class::Integral)>(&_d);
    }
    auto Float() const noexcept {
      return std::get_if<static_cast<std::size_t>(Class::Floating)>(&_d);
    }
    auto Bool() const noexcept {
      return std::get_if<static_cast<std::size_t>(Class::Boolean)>(&_d);
    }

    Data _d;
  };

  Internal _internal;

 public:
  template <typename Container>
  class JsonWrapper {
    Container *_object = nullptr;

   public:
    JsonWrapper(Container *val) : _object(val) {}

    JsonWrapper(std::nullptr_t) {}

    typename Container::iterator begin() {
      return _object ? _object->begin() : typename Container::iterator();
    }
    typename Container::iterator end() {
      return _object ? _object->end() : typename Container::iterator();
    }
    typename Container::const_iterator begin() const {
      return _object ? _object->begin() : typename Container::iterator();
    }
    typename Container::const_iterator end() const {
      return _object ? _object->end() : typename Container::iterator();
    }
  };

  template <typename Container>
  class JsonConstWrapper {
    const Container *_object = nullptr;

   public:
    JsonConstWrapper(const Container *val) : _object(val) {}

    JsonConstWrapper(std::nullptr_t) {}

    typename Container::const_iterator begin() const noexcept {
      return _object ? _object->begin() : typename Container::const_iterator();
    }
    typename Container::const_iterator end() const noexcept {
      return _object ? _object->end() : typename Container::const_iterator();
    }
  };

  Json() = default;
  Json(std::nullptr_t) {}

  explicit Json(Class type) : _internal(type) {}

  Json(std::initializer_list<Json> list) : _internal(Class::Object) {
    for (auto i = list.begin(); i != list.end(); ++i, ++i) {
      operator[](i->to_string()) = *std::next(i);
    }
  }

  template <typename T>
  explicit Json(T b, typename std::enable_if_t<std::is_same_v<T, bool>> * =
                         nullptr) noexcept
      : _internal(static_cast<bool>(b)) {}

  template <typename T>
  explicit Json(T i, typename std::enable_if_t<std::is_integral_v<T> &&
                                               !std::is_same_v<T, bool>> * =
                         nullptr) noexcept
      : _internal(static_cast<std::int64_t>(i)) {}

  template <typename T>
  explicit Json(T f, typename std::enable_if_t<std::is_floating_point_v<T>> * =
                         nullptr) noexcept
      : _internal(static_cast<double>(f)) {}

  template <typename T>
  explicit Json(T s,
                typename std::enable_if_t<std::is_convertible_v<T, std::string>>
                    * = nullptr)
      : _internal(static_cast<std::string>(s)) {}

  static Json Load(const std::string &);

  Json &operator[](const std::string &key) { return _internal.Map()[key]; }

  Json &operator[](const size_t index) {
    auto &vec = _internal.Vector();
    if (index >= vec.size()) {
      vec.resize(index + 1);
    }
    return vec[index];
  }

  Json &at(const std::string &key) { return operator[](key); }

  const Json &at(const std::string &key) const {
    return _internal.visitor<Class::Object>(
        [&](const auto &m) -> const Json & { return m.at(key); },
        []() -> const Json & {
          throw std::range_error("Not an object, no keys");
        });
  }

  Json &at(size_t index) { return operator[](index); }

  const Json &at(size_t index) const {
    return _internal.visitor<Class::Array>(
        [&](const auto &m) -> const Json & { return m.at(index); },
        []() -> const Json & {
          throw std::range_error("Not an array, no indexes");
        });
  }

  auto length() const noexcept {
    return _internal.visitor<Class::Array>(
        [&](const auto &m) { return static_cast<int>(m.size()); },
        []() { return -1; });
  }

  bool has_key(const std::string &key) const noexcept {
    return _internal.visitor<Class::Object>(
        [&](const auto &m) { return m.count(key) != 0; },
        []() { return false; });
  }

  int size() const noexcept {
    if (auto m = _internal.Map(); m != nullptr) {
      return static_cast<int>(m->size());
    }
    if (auto v = _internal.Vector(); v != nullptr) {
      return static_cast<int>(v->size());
    } else {
      return -1;
    }
  }

  Class JsonType() const noexcept { return _internal.type(); }

  bool is_null() const noexcept { return _internal.type() == Class::Null; }

  std::string to_string() const noexcept {
    return _internal.visitor<Class::String>([](const auto &o) { return o; },
                                            []() { return std::string{}; });
  }

  double to_float() const noexcept {
    return _internal.visitor<Class::Floating>([](const auto &o) { return o; },
                                              []() { return double{}; });
  }

  std::int64_t to_int() const noexcept {
    return _internal.visitor<Class::Integral>([](const auto &o) { return o; },
                                              []() { return std::int64_t{}; });
  }

  bool to_bool() const noexcept {
    return _internal.visitor<Class::Boolean>([](const auto &o) { return o; },
                                             []() { return false; });
  }

  JsonWrapper<std::map<std::string, Json>> object_range() {
    return std::get_if<static_cast<std::size_t>(Class::Object)>(&_internal._d);
  }

  JsonWrapper<std::vector<Json>> array_range() {
    return std::get_if<static_cast<std::size_t>(Class::Array)>(&_internal._d);
  }

  JsonConstWrapper<std::map<std::string, Json>> object_range() const {
    return std::get_if<static_cast<std::size_t>(Class::Object)>(&_internal._d);
  }

  JsonConstWrapper<std::vector<Json>> array_range() const {
    return std::get_if<static_cast<std::size_t>(Class::Array)>(&_internal._d);
  }

  std::string dump(long depth = 1, std::string tab = "  ") const {
    switch (_internal.type()) {
      case Class::Null:
        return "null";
      case Class::Object: {
        std::string pad = "";
        for (long i = 0; i < depth; ++i, pad += tab) {
        }
        std::string s = "{\n";
        bool skip = true;
        for (auto &p : *_internal.Map()) {
          if (!skip) {
            s += ",\n";
          }
          s += (pad + "\"" + json_escape(p.first) +
                "\" : " + p.second.dump(depth + 1, tab));
          skip = false;
        }
        s += ("\n" + pad.erase(0, 2) + "}");
        return s;
      }
      case Class::Array: {
        std::string s = "[";
        bool skip = true;
        for (auto &p : *_internal.Vector()) {
          if (!skip) {
            s += ", ";
          }
          s += p.dump(depth + 1, tab);
          skip = false;
        }
        s += "]";
        return s;
      }
      case Class::String:
        return "\"" + json_escape(*_internal.String()) + "\"";
      case Class::Floating:
        return std::to_string(*_internal.Float());
      case Class::Integral:
        return std::to_string(*_internal.Int());
      case Class::Boolean:
        return *_internal.Bool() ? "true" : "false";
    }
    throw std::runtime_error("Unhandled Json Type");
  }

 private:
  static std::string json_escape(const std::string &str) {
    std::string output;
    for (char i : str) {
      switch (i) {
        case '\"':
          output += "\\\"";
          break;
        case '\\':
          output += "\\\\";
          break;
        case '\b':
          output += "\\b";
          break;
        case '\f':
          output += "\\f";
          break;
        case '\n':
          output += "\\n";
          break;
        case '\r':
          output += "\\r";
          break;
        case '\t':
          output += "\\t";
          break;
        default:
          output += i;
          break;
      }
    }
    return output;
  }
};

struct JsonParser {
  static bool isspace(const char c) noexcept { return ::isspace(c) != 0; }

  static void consume_ws(const std::string &str, size_t offset) {
    while (isspace(str.at(offset)) && offset <= str.size()) {
      ++offset;
    }
  }

  static Json parse_object(const std::string &str, size_t &offset) {
    Json Object(Json::Class::Object);

    ++offset;
    consume_ws(str, offset);
    if (str.at(offset) == '}') {
      ++offset;
      return Object;
    }

    for (; offset < str.size();) {
      Json key = parse_next(str, offset);
      consume_ws(str, offset);
      if (str.at(offset) != ':') {
        throw std::runtime_error(
            std::string("[Json Error] Object: Expected ':' , found '") +
            str.at(offset) + "'\n");
      }
      consume_ws(str, ++offset);
      Json value = parse_next(str, offset);
      Object[key.to_string()] = value;

      consume_ws(str, offset);
      if (str.at(offset) == ',') {
        ++offset;
        continue;
      } else if (str.at(offset) == '}') {
        ++offset;
        break;
      } else {
        throw std::runtime_error(
            std::string("[Json Error] Object: Expected ',' or '}' , found '") +
            str.at(offset) + "'\n");
      }
    }
    return Object;
  }

  static Json parse_array(const std::string &str, size_t &offset) {
    Json Array(Json::Class::Array);
    size_t index = 0;

    ++offset;
    consume_ws(str, offset);
    if (str.at(offset) == ']') {
      ++offset;
      return Array;
    }

    for (; offset < str.size();) {
      Array[index++] = parse_next(str, offset);
      consume_ws(str, offset);

      if (str.at(offset) == ',') {
        ++offset;
        continue;
      } else if (str.at(offset) == ']') {
        ++offset;
        break;
      } else {
        throw std::runtime_error(
            std::string("[Json Error] Array: Expected ',' or ']', found '") +
            str.at(offset) + "'\n");
      }
    }
    return Array;
  }

  static Json parse_string(const std::string &str, size_t &offset) {
    std::string val;
    for (char c = str.at(++offset); c != '\"'; c = str.at(++offset)) {
      if (c == '\\') {
        switch (str.at(++offset)) {
          case '\"':
            val += '\"';
            break;
          case '\\':
            val += '\\';
            break;
          case '/':
            val += '/';
            break;
          case 'b':
            val += '\b';
            break;
          case 'f':
            val += '\f';
            break;
          case 'n':
            val += '\n';
            break;
          case 'r':
            val += '\r';
            break;
          case 't':
            val += '\t';
            break;
          case 'u': {
            val += "\\u";
            for (size_t i = 1; i <= 4; ++i) {
              c = str.at(offset + i);
              if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                  (c >= 'A' && c <= 'F')) {
                val += c;
              } else {
                throw std::runtime_error(
                    std::string("[Json Error] String:Expected hex character in "
                                "unicode escape, found '") +
                    c + "'\n");
              }
            }
            offset += 4;
          } break;
          default:
            val += '\\';
            break;
        }
      } else {
        val += c;
      }
    }
    ++offset;
    return Json(val);
  }

  static Json parse_number(const std::string &str, size_t &offset) {
    std::string val, exp_str;
    char c = '\0';
    bool is_double = false;
    bool is_negative = false;
    bool is_exp_negative = false;
    std::int64_t exp = 0;
    if (offset < str.size() && str.at(offset) == '-') {
      is_negative = true;
      ++offset;
    }

    for (; offset < str.size();) {
      c = str.at(offset++);
      if (c >= '0' && c <= '9') {
        val += c;
      } else if (c == '.' && !is_double) {
        val += c;
        is_double = true;
      } else {
        break;
      }
    }

    if (offset < str.size() && (c == 'E' || c == 'e')) {
      c = str.at(offset++);
      if (c == '-') {
        is_exp_negative = true;
      } else if (c == '+') {
      } else {
        --offset;
      }

      for (; offset < str.size();) {
        c = str.at(offset++);
        if (c >= '0' && c <= '9') {
          exp_str += c;
        } else if (!isspace(c) && c != ',' && c != ']' && c != '}') {
          throw std::runtime_error(
              std::string("[Json Error] Number: Expected a "
                          "number for exponent, found '") +
              c + "'\n");
        } else {
          break;
        }
      }
      exp = domino::detail::parse_num<std::int64_t>(exp_str) *
            (is_exp_negative ? -1 : 1);
    } else if (offset < str.size() &&
               (!isspace(c) && c != ',' && c != ']' && c != '}')) {
      throw std::runtime_error(
          std::string("[Json Error] Number: Unexpected character '") + c +
          "'\n");
    }
    --offset;

    if (is_double) {
      return Json((is_negative ? -1 : 1) *
                  domino::detail::parse_num<double>(val) *
                  std::pow(10, exp));
    } else {
      if (!exp_str.empty()) {
        return Json((is_negative ? -1 : 1) *
                    static_cast<double>(
                        domino::detail::parse_num<std::int64_t>(val)) *
                    std::pow(10, exp));
      } else {
        return Json((is_negative ? -1 : 1) *
                    domino::detail::parse_num<std::int64_t>(val));
      }
    }
  }

  static Json parse_bool(const std::string &str, size_t &offset) {
    if (str.substr(offset, 4) == "true") {
      offset += 4;
      return Json(true);
    } else if (str.substr(offset, 5) == "false") {
      offset += 5;
      return Json(false);
    } else {
      throw std::runtime_error(
          std::string(
              "[Json Error] Bool: Expected 'true' or 'false', found '") +
          str.substr(offset, 5) + "'\n");
    }
  }

  static Json parse_null(const std::string &str, size_t &offset) {
    if (str.substr(offset, 4) != "null") {
      throw std::runtime_error(
          std::string("[Json Error] Null: Expected 'null', found '") +
          str.substr(offset, 4) + "'\n");
    }
    offset += 4;
    return Json();
  }

  static Json parse_next(const std::string &str, size_t &offset) {
    if (str.size() == 0) {
      throw std::runtime_error(
          std::string("[Json Error] Parse: param str is empty'") + str + "'\n");
    }
    consume_ws(str, offset);
    char value = str.at(offset);
    switch (value) {
      case '[':
        return parse_array(str, offset);
      case '{':
        return parse_object(str, offset);
      case '\"':
        return parse_string(str, offset);
      case 't':
      case 'f':
        return parse_bool(str, offset);
      case 'n':
        return parse_null(str, offset);
      default:
        if ((value <= '9' && value >= '0') || value == '-') {
          return parse_number(str, offset);
        }
        throw std::runtime_error(
            std::string("[Json Error] Parse: Unexpected starting character '") +
            value + "'\n");
    }
  }
};

inline Json Json::Load(const std::string &str) {
  size_t offset = 0;
  return JsonParser::parse_next(str, offset);
}

}  // namespace domino

#endif  // DOMINO_UTIL_JSON_H_