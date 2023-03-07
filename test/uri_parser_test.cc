#include <iostream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "uri_parser.h"

void test_base() {
  using namespace domino::http;
  
  URIParser uri;

  parse("https://john.doe:pass@www.example.com:123/forum/questions/?tag=networking&order=newest#top", uri);
  std::cout << uri.scheme << std::endl;
  std::cout << uri.userinfo << std::endl;
  std::cout << uri.host << std::endl;
  std::cout << uri.port << std::endl;
  std::cout << uri.path << std::endl;
  std::cout << uri.query << std::endl;
  std::cout << uri.fragment << std::endl;
}

int main(int argc, char **argv) {
  test_base();
  return 0;
}