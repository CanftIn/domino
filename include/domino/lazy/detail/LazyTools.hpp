#ifndef DOMINO_LAZY_DETAIL_LAZYTOOLS_HPP_
#define DOMINO_LAZY_DETAIL_LAZYTOOLS_HPP_

#include <iterator>

namespace domino::lazy {

template <class I>
concept BasicIterable = requires(I i) {
  { std::begin(i) } -> std::input_or_output_iterator;
  { std::end(i) } -> std::input_or_output_iterator;
};

}



#endif  // DOMINO_LAZY_DETAIL_LAZYTOOLS_HPP_