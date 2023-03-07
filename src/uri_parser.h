#ifndef DOMINO_URI_PARSER_H_
#define DOMINO_URI_PARSER_H_

#include <stdint.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace domino {
namespace http {
//                     hierarchical part
//         ┌───────────────────┴─────────────────────┐
//                     authority               path
//         ┌───────────────┴───────────────┐┌───┴────┐
//   abc://username:password@example.com:123/path/data?key=value&key2=value2#fragid1
//   └┬┘   └───────┬───────┘ └────┬────┘ └┬┘           └─────────┬─────────┘ └──┬──┘
// scheme  user information     host     port                  query         fragment
// 
//   urn:example:mammal:monotreme:echidna
//   └┬┘ └──────────────┬───────────────┘
// scheme              path

class URIParser {
 public:
  URIParser() { init(); }

  virtual ~URIParser() {}

  URIParser(const URIParser& uri) { copy(uri); }

  URIParser& operator=(const URIParser& uri) {
    if (this != &uri) {
      copy(uri);
    }
    return *this;
  }

  URIParser(URIParser&& uri) {
    if (this != &uri) {
      copy(uri);
      uri.init();
    }
  }

  URIParser& operator=(URIParser&& uri) {
    if (this != &uri) {
      copy(uri);
      uri.init();
    }
    return *this;
  }

 private:
  void init() {
    scheme = "";
    userinfo = "";
    host = "";
    port = "";
    path = "";
    query = "";
    fragment = "";
    state = URI_STATE::INIT;
    error = 0;
  }

  void copy(const URIParser& uri);

  std::string to_string() const;

  friend int parse(const char* str, URIParser& uri);

  friend int parse(const std::string& str, URIParser& uri) {
    return parse(str.c_str(), uri);
  }

  static std::vector<std::string> split_path(const std::string& path);
 
 public:
  enum class URI_STATE : unsigned int {
    INIT,
    SUCCESS,
    INVALID,
    ERROR
  };

 public:
  std::string scheme;
  std::string userinfo;
  std::string host;
  std::string port;
  std::string path;
  std::string query;
  std::string fragment;
  URI_STATE state;
  int error;
};

}  // namespace http
}  // namespace domino

#endif  // DOMINO_URI_PARSER_H_
