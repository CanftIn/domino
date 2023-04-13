#ifndef DOMINO_UTIL_ITERATOR_RANGE_H_
#define DOMINO_UTIL_ITERATOR_RANGE_H_

#include <utility>

namespace domino {

template <typename IteratorT>
class iterator_range {
  IteratorT begin_iterator, end_iterator;

 public:
  template <typename Container>
  iterator_range(Container &&c)
      : begin_iterator(c.begin()), end_iterator(c.end()) {}
  iterator_range(IteratorT begin_iterator, IteratorT end_iterator)
      : begin_iterator(std::move(begin_iterator)),
        end_iterator(std::move(end_iterator)) {}

  IteratorT begin() const { return begin_iterator; }
  IteratorT end() const { return end_iterator; }
  bool empty() const { return begin_iterator == end_iterator; }
};

template <class T>
iterator_range<T> make_range(T x, T y) {
  return iterator_range<T>(std::move(x), std::move(y));
}

template <typename T>
iterator_range<T> make_range(std::pair<T, T> p) {
  return iterator_range<T>(std::move(p.first), std::move(p.second));
}

}  // namespace domino

#endif  // DOMINO_UTIL_ITERATOR_RANGE_H_