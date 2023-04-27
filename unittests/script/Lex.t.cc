#include <domino/script/Lexer.h>
#include <domino/support/raw_ostream.h>
#include <domino/util/StringRef.h>
#include <domino/util/ToString.h>

int main() {
  domino::StringRef str(R"(
# this is a comment
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
}
)");

  std::string hhh = domino::to_string(str);
  domino::script::LexerBuffer lexer(str.begin(), str.end(), std::string(str));

  lexer.getNextToken();  // prime the lexer

  bool is_next = true;
  while (is_next) {
    domino::script::Token tok = lexer.getCurToken();
    if (tok == domino::script::tok_eof) {
      domino::errs() << tok << "\n";
      is_next = false;
    } else {
      domino::errs() << tok << "\n";
      lexer.consume(tok);
    }
  }

  return 0;
}