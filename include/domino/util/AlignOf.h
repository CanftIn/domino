#ifndef DOMINO_UTIL_ALIGNOF_H_
#define DOMINO_UTIL_ALIGNOF_H_

#include <type_traits>

namespace domino {

template <typename T, typename... Ts>
struct AlignedCharArrayUnion {
  using AlignedUnion = std::aligned_union_t<1, T, Ts...>;
  alignas(alignof(AlignedUnion)) char buffer[sizeof(AlignedUnion)];
};

}  // namespace domino

#endif  // DOMINO_UTIL_ALIGNOF_H_