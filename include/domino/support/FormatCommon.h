#ifndef DOMINO_SUPPORT_FORMATCOMMON_H_
#define DOMINO_SUPPORT_FORMATCOMMON_H_

#include <domino/support/FormatVariadicDetails.h>
#include <domino/support/raw_ostream.h>
#include <domino/util/SmallString.h>

namespace domino {

enum class AlignStyle { Left, Center, Right };

struct FmtAlign {
  detail::format_adapter& Adapter;
  AlignStyle Where;
  size_t Amount;
  char Fill;

  FmtAlign(detail::format_adapter& Adapter, AlignStyle Where, size_t Amount,
           char Fill = ' ')
      : Adapter(Adapter), Where(Where), Amount(Amount), Fill(Fill) {}

  void format(raw_ostream& S, StringRef Options) {
    if (Amount == 0) {
      Adapter.format(S, Options);
      return;
    }
    SmallString<64> Item;
    raw_svector_ostream Stream(Item);

    Adapter.format(Stream, Options);
    if (Amount <= Item.size()) {
      S << Item;
      return;
    }

    size_t PadAmount = Amount - Item.size();
    switch (Where) {
      case AlignStyle::Left:
        S << Item;
        fill(S, PadAmount);
        break;
      case AlignStyle::Center: {
        size_t X = PadAmount / 2;
        fill(S, X);
        S << Item;
        fill(S, PadAmount - X);
        break;
      }
      default:
        fill(S, PadAmount);
        S << Item;
        break;
    }
  }

 private:
  void fill(raw_ostream& S, uint32_t Count) {
    for (uint32_t I = 0; I < Count; ++I) S << Fill;
  }
};

}  // namespace domino

#endif  // DOMINO_SUPPORT_FORMATCOMMON_H_