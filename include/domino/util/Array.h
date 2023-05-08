#ifndef DOMINO_UTIL_ARRAY_H_
#define DOMINO_UTIL_ARRAY_H_

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>

namespace domino {

namespace detail {

template <typename Tp, size_t N>
struct array_traits final {
  using Type = Tp[N];

  static constexpr Tp& ref(const Type& T, size_t n) noexcept {
    return const_cast<Tp&>(T[n]);
  }

  static constexpr Tp* ptr(const Type& T) noexcept {
    return const_cast<Tp*>(T);
  }
};

template <typename Tp>
struct array_traits<Tp, 0> final {
  struct Type final {};

  static constexpr Tp& ref(const Type& T, std::size_t) noexcept {
    return *ptr(T);
  }

  static constexpr Tp* ptr(const Type&) noexcept { return nullptr; }
};

}  // namespace detail

template <typename Tp, size_t N>
class Array final {
 public:
  using value_type = Tp;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using iterator = pointer;
  using const_iterator = const_pointer;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 private:
  using ArrType = detail::array_traits<Tp, N>;

 public:
  typename ArrType::Type Elems;

 public:
  constexpr void fill(const value_type& V) { std::fill_n(begin(), size(), V); }

  template <size_t M, typename std::enable_if<(M <= N), int>::type = 0>
  void swap(Array<Tp, M>& other) noexcept {
    std::swap_ranges(begin(), begin() + M, other.begin());
  }

  constexpr iterator begin() noexcept { return iterator(data()); }

  constexpr const_iterator begin() const noexcept {
    return const_iterator(data());
  }

  constexpr iterator end() noexcept { return iterator(data() + size()); }

  constexpr const_iterator end() const noexcept {
    return const_iterator(data() + size());
  }

  constexpr reverse_iterator rbegin() noexcept {
    return reverse_iterator(end());
  }

  constexpr const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }

  constexpr reverse_iterator rend() noexcept {
    return reverse_iterator(begin());
  }

  constexpr const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }

  constexpr const_iterator cbegin() const noexcept {
    return const_iterator(data());
  }

  constexpr const_iterator cend() const noexcept {
    return const_iterator(data() + size());
  }

  constexpr const_reverse_iterator crbegin() const noexcept {
    return const_reverse_iterator(end());
  }

  constexpr const_reverse_iterator crend() const noexcept {
    return const_reverse_iterator(begin());
  }

  constexpr size_type size() const noexcept { return N; }

  constexpr size_type max_size() const noexcept { return N; }

  constexpr bool empty() const noexcept { return N == 0; }

  constexpr reference operator[](size_type n) noexcept {
    return ArrType::ref(Elems, n);
  }

  constexpr const_reference operator[](size_type n) const noexcept {
    return ArrType::ref(Elems, n);
  }

  constexpr reference at(size_type n) {
    if (n >= size()) {
      throw std::out_of_range("Array::at");
    }
    return ArrType::ref(Elems, n);
  }

  constexpr const_reference at(size_type n) const {
    return n < N ? ArrType::ref(Elems, n)
                 : throw std::out_of_range("Array::at"),
           ArrType::ref(Elems, n);
  }

  constexpr reference front() noexcept { return *begin(); }

  constexpr const_reference front() const noexcept { return *begin(); }

  constexpr reference back() noexcept { return N ? *(end() - 1) : *end(); }

  constexpr const_reference back() const noexcept {
    return N ? ArrType::ref(Elems, N - 1) : ArrType::ref(Elems, N);
  }

  constexpr pointer data() noexcept { return ArrType::ptr(Elems); }

  constexpr const_pointer data() const noexcept { return ArrType::ptr(Elems); }
};

template <typename Tp, typename... Up>
Array(Tp, Up...)
    -> Array<std::enable_if_t<(std::is_same<Tp, Up>::value && ...), Tp>,
             1 + sizeof...(Up)>;

namespace detail {

template <class T, size_t N>
constexpr inline bool array_equals(const Array<T, N>& lhs,
                                   const Array<T, N>& rhs,
                                   size_t current_index) noexcept {
  return (current_index == N)
             ? true
             : (lhs.at(current_index) == rhs.at(current_index) &&
                array_equals(lhs, rhs, current_index + 1));
}

template <class T, size_t N>
constexpr inline bool array_less(const Array<T, N>& lhs, const Array<T, N>& rhs,
                                 size_t current_index) noexcept {
  return (current_index == N)
             ? false
             : (lhs.at(current_index) < rhs.at(current_index) ||
                array_less(lhs, rhs, current_index + 1));
}

}  // namespace detail

template <class T, size_t N>
constexpr inline bool operator==(const Array<T, N>& one,
                                 const Array<T, N>& two) {
  return detail::array_equals(one, two, 0);
}

template <class T, size_t N>
constexpr inline bool operator!=(const Array<T, N>& one,
                                 const Array<T, N>& two) {
  return !(one == two);
}

template <class T, size_t N>
constexpr inline bool operator<(const Array<T, N>& one,
                                const Array<T, N>& two) {
  return detail::array_less(one, two, 0);
}

template <class T, size_t N>
constexpr inline bool operator>(const Array<T, N>& one,
                                const Array<T, N>& two) {
  return two < one;
}

template <class T, size_t N>
constexpr inline bool operator<=(const Array<T, N>& one,
                                 const Array<T, N>& two) {
  return !(one > two);
}

template <class T, size_t N>
constexpr inline bool operator>=(const Array<T, N>& one,
                                 const Array<T, N>& two) {
  return !(one < two);
}

template <typename Tp, size_t N>
inline void swap(Array<Tp, N>& one, Array<Tp, N>& two) noexcept {
  one.swap(two);
}

template <size_t Int, typename Tp, size_t N>
constexpr Tp& get(Array<Tp, N>& A) noexcept {
  static_assert(Int < N, "Index out of bounds in domino::get");
  return detail::array_traits<Tp, N>::ref(A.Elems, Int);
}

template <size_t Int, typename Tp, size_t N>
constexpr Tp&& get(Array<Tp, N>&& A) noexcept {
  static_assert(Int < N, "Index out of bounds in domino::get");
  return std::move(get<Int>(A));
}

template <size_t Int, typename Tp, size_t N>
constexpr const Tp& get(const Array<Tp, N>& A) noexcept {
  static_assert(Int < N, "Index out of bounds in domino::get");
  return detail::array_traits<Tp, N>::ref(A.Elems, Int);
}

namespace detail {

template <class T, size_t N, size_t... Index>
constexpr inline Array<T, N - 1> tail_(const Array<T, N>& A,
                                       std::index_sequence<Index...>) {
  static_assert(sizeof...(Index) == N - 1, "invariant");
  return {{get<Index + 1>(A)...}};
}

template <class T, size_t N, size_t... Index>
constexpr inline Array<T, N + 1> prepend_(T&& head, const Array<T, N>& tail,
                                          std::index_sequence<Index...>) {
  return {{std::forward<T>(head), get<Index>(tail)...}};
}

template <class T, size_t N, size_t... Index>
constexpr Array<T, N> to_array_(const T (&arr)[N],
                                std::index_sequence<Index...>) {
  return {{arr[Index]...}};
}

}  // namespace detail

template <class T, size_t N>
constexpr inline Array<T, N - 1> tail(const Array<T, N>& A) {
  static_assert(N > 0,
                "Can only call tail() on an array with at least one element");
  return detail::tail_(A, std::make_index_sequence<N - 1>());
}

template <class T, size_t N>
constexpr inline Array<T, N + 1> prepend(T&& head, const Array<T, N>& tail) {
  return detail::prepend_(std::forward<T>(head), tail,
                          std::make_index_sequence<N>());
}

template <class T, size_t N>
constexpr Array<T, N> to_array(const T (&arr)[N]) {
  return detail::to_array_(arr, std::make_index_sequence<N>());
}

}  // namespace domino

#endif  // DOMINO_UTIL_ARRAY_H_