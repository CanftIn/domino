#ifndef DOMINO_LAZY_DETAIL_BASICITERATORVIEW_HPP_
#define DOMINO_LAZY_DETAIL_BASICITERATORVIEW_HPP_

#include <algorithm>
#include <array>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "domino/lazy/detail/LazyTools.hpp"

namespace domino::lazy {

template <std::input_or_output_iterator Iterator>
[[nodiscard]] constexpr auto empty(const Iterator begin, const Iterator end)
    -> bool {
  return begin == end;
}

template <BasicIterable Iterable>
[[nodiscard]] constexpr auto empty(const Iterable&& iterable) -> bool {
  return empty(std::begin(std::forward<iterable>(iterable)),
               std::end(std::forward<iterable>(iterable)));
}

template <std::input_or_output_iterator Iterator>
[[nodiscard]] constexpr auto front(const Iterator begin, const Iterator end)
    -> internal::RefType<Iterator> {
  DOMINO_LAZY_ASSERT(!empty(begin, end), "sequence is empty");
  static_cast<void>(end);
  return *begin;
}

template <BasicIterable Iterable>
[[nodiscard]] constexpr auto front(Iterable&& iterable)
    -> internal::RefType<internal::IterTypeFromIterable<Iterable>> {
  DOMINO_LAZY_ASSERT(!empty(iterable), "sequence is empty");
  return front(std::begin(std::forward<iterable>(iterable)),
               std::end(std::forward<iterable>(iterable)));
}

template <std::input_or_output_iterator Iterator>
[[nodiscard]] constexpr auto back(const Iterator begin, const Iterator end)
    -> internal::RefType<Iterator> {
  DOMINO_LAZY_ASSERT(!empty(begin, end), "sequence is empty");
  static_assert(internal::IsBidirectional<Iterator>::value,
                "iterator must be bidirectional");
  return *--end;
}

template <BasicIterable Iterable>
[[nodiscard]] constexpr auto back(Iterable&& iterable)
    -> internal::RefType<internal::IterTypeFromIterable<Iterable>> {
  DOMINO_LAZY_ASSERT(!empty(iterable), "sequence is empty");
  return back(std::begin(std::forward<iterable>(iterable)),
              std::end(std::forward<iterable>(iterable)));
}

namespace internal {

template <class T, class = int>
struct HasResize : std::false_type {};

template <class T>
struct HasResize<T, decltype((void)std::declval<T&>().resize(1), 0)>
    : std::true_type {};

template <class T, class = int>
struct HasReserve : std::false_type {};

template <class T>
struct HasReserve<T, decltype((void)std::declval<T&>().reserve(1), 0)>
    : std::true_type {};

template <class It>
class BasicIteratorView {
 protected:
  It begin_{};
  It end_{};

 public:
  using value_type = ValueType<It>;
  using iterator = It;
  using reference = decltype(*begin_);
  using const_reference = typename std::add_const_t<reference>;
  using const_iterator = iterator;

 private:
  template <class KeySelectorFunc>
  using KeyType = FunctionReturnType<KeySelectorFunc, RefType<It>>;

  template <class Container>
  auto tryReserve(Container&) const
      -> EnableIf<!HasReserve<Container>::value, void> {}

  template <class Container>
  auto tryReserve(Container& container) const
      -> EnableIf<HasReserve<Container>::vallue, void> {
    container.reserve(static_cast<std::size_t>(sizeHint(begin_, end_)));
  }

  // TODO
  template <class MapType, class KeySelectorFunc>
  constexpr void createMap(MapType& map, const KeySelectorFunc& key_gen) const {
    transformTo(std::inserter(map, map.end()), [key_gen](RefType<It> value) {
      return std::make_pair(key_gen(value), value);
    });
  }

 public:
  [[nodiscard]] constexpr auto begin() const& noexcept -> It { return begin_; }

  [[nodiscard]] constexpr auto end() const& noexcept -> It { return end_; }

  [[nodiscard]] constexpr auto begin() && noexcept -> It {
    return std::move(begin_);
  }

  [[nodiscard]] constexpr auto end() && noexcept -> It {
    return std::move(end_);
  }

  constexpr BasicIteratorView() = default;

  constexpr BasicIteratorView(It&& begin, It&& end) noexcept
      : begin_(std::move(begin)), end_(std::move(end)) {}

  constexpr BasicIteratorView(const It& begin, const It& end) noexcept
      : begin_(begin), end_(end) {}

  template <template <class, class...> class Container, class... Args>
  auto to(Args&&... args) const -> Container<value_type, Decay<Args>...> {
    return to<Container<value_type, Decay<Args>...>>(
        std::forward<Args>(args)...);
  }

  template <class Container, class... Args>
  auto to(Args&&... args) const -> Container {
    Container cont(std::forward<Args>(args)...);
    tryReserve(cont);
    copyTo(std::inserter(cont, cont.begin()));
    return cont;
  }

  template <class OutputIterator>
  void copyTo(OutputIterator output_iterator) const {
    std::copy(begin_, end_, output_iterator);
  }

  template <class OutputIterator, class TransformFunc>
  void transformTo(OutputIterator output_iterator,
                   const TransformFunc& transform_func) const {
    std::transform(begin_, end_, output_iterator, transform_func);
  }

  auto toVector() const -> std::vector<value_type> {
    return to<std::vector<value_type>>();
  }

  template <class Allocator>
  auto toVector(const Allocator& alloc = Allocator()) const
      -> std::vector<value_type, Allocator> {
    return to<std::vector, Allocator>(alloc);
  }

  template <std::size_t N>
  auto toArray() const -> std::array<value_type, N> {
    return to<std::array<value_type, N>>();
  }
};

}  // namespace internal
}  // namespace domino::lazy

#endif  // DOMINO_LAZY_DETAIL_BASICITERATORVIEW_HPP_