#include <domino/http/uri_parser.h>

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

TEST(URIParser, Base) {
  using namespace domino::http;

  URIParser uri;

  parse(
      "https://john.doe:pass@www.example.com:123/forum/questions/"
      "?tag=networking&order=newest#top",
      uri);
  std::cout << uri.scheme << std::endl;
  std::cout << uri.userinfo << std::endl;
  std::cout << uri.host << std::endl;
  std::cout << uri.port << std::endl;
  std::cout << uri.path << std::endl;
  std::cout << uri.query << std::endl;
  std::cout << uri.fragment << std::endl;

  parse("ldap://user@[2001:db8::7]:12345/c=GB?objectClass?one", uri);
  std::cout << uri.scheme << std::endl;
  std::cout << uri.userinfo << std::endl;
  std::cout << uri.host << std::endl;
  std::cout << uri.port << std::endl;
  std::cout << uri.path << std::endl;
  std::cout << uri.query << std::endl;
  std::cout << uri.fragment << std::endl;
}