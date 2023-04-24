#ifndef DOMINO_UTIL_EDITDISTANCE_H_
#define DOMINO_UTIL_EDITDISTANCE_H_

#include <domino/util/ArrayRef.h>

#include <algorithm>
#include <memory>

namespace domino {

template <typename T, typename Functor>
unsigned ComputeMappedEditDistance(ArrayRef<T> FromArray, ArrayRef<T> ToArray,
                                   Functor Map, bool AllowReplacements = true,
                                   unsigned MaxEditDistance = 0) {
  // The algorithm implemented below is the "classic"
  // dynamic-programming algorithm for computing the Levenshtein
  // distance, which is described here:
  //
  //   http://en.wikipedia.org/wiki/Levenshtein_distance
  //
  // Although the algorithm is typically described using an m x n
  // array, only one row plus one element are used at a time, so this
  // implementation just keeps one vector for the row.  To update one entry,
  // only the entries to the left, top, and top-left are needed.  The left
  // entry is in Row[x-1], the top entry is what's in Row[x] from the last
  // iteration, and the top-left entry is stored in Previous.
  typename ArrayRef<T>::size_type m = FromArray.size();
  typename ArrayRef<T>::size_type n = ToArray.size();

  if (MaxEditDistance) {
    // If the difference in size between the 2 arrays is larger than the max
    // distance allowed, we can bail out as we will always need at least
    // MaxEditDistance insertions or removals.
    typename ArrayRef<T>::size_type AbsDiff = m > n ? m - n : n - m;
    if (AbsDiff > MaxEditDistance) return MaxEditDistance + 1;
  }

  const unsigned SmallBufferSize = 64;
  unsigned SmallBuffer[SmallBufferSize];
  std::unique_ptr<unsigned[]> Allocated;
  unsigned *Row = SmallBuffer;
  if (n + 1 > SmallBufferSize) {
    Row = new unsigned[n + 1];
    Allocated.reset(Row);
  }

  for (unsigned i = 1; i <= n; ++i) Row[i] = i;

  for (typename ArrayRef<T>::size_type y = 1; y <= m; ++y) {
    Row[0] = y;
    unsigned BestThisRow = Row[0];

    unsigned Previous = y - 1;
    const auto &CurItem = Map(FromArray[y - 1]);
    for (typename ArrayRef<T>::size_type x = 1; x <= n; ++x) {
      int OldRow = Row[x];
      if (AllowReplacements) {
        Row[x] = std::min(Previous + (CurItem == Map(ToArray[x - 1]) ? 0u : 1u),
                          std::min(Row[x - 1], Row[x]) + 1);
      } else {
        if (CurItem == Map(ToArray[x - 1]))
          Row[x] = Previous;
        else
          Row[x] = std::min(Row[x - 1], Row[x]) + 1;
      }
      Previous = OldRow;
      BestThisRow = std::min(BestThisRow, Row[x]);
    }

    if (MaxEditDistance && BestThisRow > MaxEditDistance)
      return MaxEditDistance + 1;
  }

  unsigned Result = Row[n];
  return Result;
}

template <typename T>
unsigned ComputeEditDistance(ArrayRef<T> FromArray, ArrayRef<T> ToArray,
                             bool AllowReplacements = true,
                             unsigned MaxEditDistance = 0) {
  return ComputeMappedEditDistance(
      FromArray, ToArray, [](const T &X) -> const T & { return X; },
      AllowReplacements, MaxEditDistance);
}

}  // namespace domino

#endif  // DOMINO_UTIL_EDITDISTANCE_H_