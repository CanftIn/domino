#ifndef DOMINO_SCRIPT_LEXER_H_
#define DOMINO_SCRIPT_LEXER_H_

#include <domino/util/StringRef.h>

#include <memory>
#include <string>

namespace domino {

namespace script {

struct Location {
  std::shared_ptr<std::string> file;
  int line;
  int col;
};

enum Token : int {
  tok_semicolon = ';',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',

  tok_eof = -1,

  tok_return = -2,
  tok_var = -3,
  tok_def = -4,

  tok_identifier = -5,
  tok_number = -6,
};

class Lexer {
 public:
  Lexer(std::string filename)
      : curLocation{std::make_shared<std::string>(filename), 0, 0} {}

  virtual ~Lexer() = default;

  Token getCurToken() const { return curTok; }

  Token getNextToken() { return curTok = getTok(); }

  void consume(Token tok) {
    assert(curTok == tok && "consume() called with wrong token");
    getNextToken();
  }

  StringRef getId() {
    assert(curTok == Token::tok_identifier &&
           "getId() called with wrong token");
    return identifierStr;
  }

  double getValue() {
    assert(curTok == Token::tok_number && "getValue() called with wrong token");
    return numVal;
  }

  Location getLocation() const { return curLocation; }

  int getLine() const { return curLineNum; }

  int getCol() const { return curCol; }

 private:
  virtual domino::StringRef readNextLine() = 0;

  int getNextChar() {
    if (curLineBuffer.empty()) return EOF;
    ++curCol;
    auto nextChar = curLineBuffer.front();
    curLineBuffer = curLineBuffer.drop_front();
    if (curLineBuffer.empty()) curLineBuffer = readNextLine();
    if (nextChar == '\n') {
      ++curLineNum;
      curCol = 0;
    }
    return nextChar;
  }

  Token getTok() {
    // Skip any whitespace.
    while (isspace(lastChar)) lastChar = Token(getNextChar());

    curLocation.col = curCol;
    curLocation.line = curLineNum;

    if (isalpha(lastChar)) {  // identifier: [a-zA-Z][a-zA-Z0-9]*
      identifierStr = (char)lastChar;
      while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
        identifierStr += lastChar;

      if (identifierStr == "return") return tok_return;
      if (identifierStr == "var") return tok_var;
      if (identifierStr == "def") return tok_def;
      return tok_identifier;
    }

    if (isdigit(lastChar) || lastChar == '.') {  // Number: [0-9.]+
      std::string numStr;
      do {
        numStr += lastChar;
        lastChar = Token(getNextChar());
      } while (isdigit(lastChar) || lastChar == '.');

      numVal = strtod(numStr.c_str(), 0);
      return tok_number;
    }

    if (lastChar == '#') {
      // Comment until end of line.
      do lastChar = Token(getNextChar());
      while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

      if (lastChar != EOF) return getTok();
    }

    if (lastChar == EOF) return tok_eof;

    Token thisChar = Token(lastChar);
    lastChar = Token(getNextChar());
    return thisChar;
  }

  Token curTok = tok_eof;

  Location curLocation;

  std::string identifierStr;

  double numVal = 0.0;

  Token lastChar = Token(' ');

  int curLineNum = 0;

  int curCol = 0;

  domino::StringRef curLineBuffer = "\n";
};

class LexerBuffer final : public Lexer {
 public:
  LexerBuffer(const char* begin, const char* end, std::string filename)
      : Lexer(filename), current(begin), end(end) {}

 private:
  domino::StringRef readNextLine() override {
    auto *begin = current;
    while (current <= end && *current && *current != '\n') ++current;
    if (current <= end && *current) ++current;
    StringRef result{begin, static_cast<size_t>(current - begin)};
    return result;
  }

  const char *current, *end;
};

}  // namespace script

}  // namespace domino

#endif  // DOMINO_SCRIPT_LEXER_H_