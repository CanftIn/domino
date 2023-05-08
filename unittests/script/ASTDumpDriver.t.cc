#include <domino/script/Parser.h>

#include <fstream>
#include <iostream>
#include <string>

#include <domino/util/StringRef.h>

std::unique_ptr<domino::script::ModuleAST> parseInputFile(::domino::StringRef buffer) {
  domino::script::LexerBuffer lexer(buffer.begin(), buffer.end(), std::string("null"));
  domino::script::Parser parser(lexer);
  return parser.parseModule();
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return 1;
  }

  std::string filename = argv[1];
  std::ifstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
    return 1;
  }

  std::string line;
  std::string buffer;
  while (std::getline(file, line)) {
    buffer += line + "\n";
  }
  std::cout << buffer << std::endl;

  file.close();
  auto moduleAST = parseInputFile(::domino::StringRef(buffer));
  if (!moduleAST) return 1;

  domino::script::dump(*moduleAST);
  return 0;
}
