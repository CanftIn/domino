#include <domino/util/StringExtras.h>

namespace domino {

std::pair<StringRef, StringRef> getToken(StringRef Source,
                                         StringRef Delimiters) {
  // Figure out where the token starts.
  StringRef::size_type Start = Source.find_first_not_of(Delimiters);

  // Find the next occurrence of the delimiter.
  StringRef::size_type End = Source.find_first_of(Delimiters, Start);

  return std::make_pair(Source.slice(Start, End), Source.substr(End));
}

/// SplitString - Split up the specified string according to the specified
/// delimiters, appending the result fragments to the output list.
void SplitString(StringRef Source, SmallVectorImpl<StringRef> &OutFragments,
                 StringRef Delimiters) {
  std::pair<StringRef, StringRef> S = domino::getToken(Source, Delimiters);
  while (!S.first.empty()) {
    OutFragments.push_back(S.first);
    S = getToken(S.second, Delimiters);
  }
}

} // namespace domino