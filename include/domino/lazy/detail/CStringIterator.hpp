#ifndef DOMINO_LAZY_DETAIL_CSTRINGITERATOR_HPP_
#define DOMINO_LAZY_DETAIL_CSTRINGITERATOR_HPP_

#include <domino/lazy/detail/LazyTools.hpp>

namespace domino::lazy::internal {

template <class C, bool IsRandomAccess>
class CStringIterator {
 public:
  using iterator_category =
      Conditional<IsRandomAccess, std::random_access_iterator_tag,
                  std::forward_iterator_tag>;
  using value_type = C;
  using difference_type = std::ptrdiff_t;
  using pointer = const C*;
  using reference = const C&;

  constexpr CStringIterator() noexcept = default;

  constexpr CStringIterator(const C* it) noexcept : it_(it) {}

  [[nodiscard]] constexpr auto operator*() const noexcept -> reference {
    return *it_;
  }

  [[nodiscard]] constexpr auto operator->() const noexcept -> pointer {
    return it_;
  }

  constexpr auto operator++() noexcept -> CStringIterator& {
    ++it_;
    return *this;
  }

  constexpr auto operator++(int) noexcept -> CStringIterator {
    auto tmp = *this;
    ++*this;
    return tmp;
  }

  [[nodiscard]] constexpr friend auto operator!=(
      const CStringIterator& a, const CStringIterator& b) noexcept -> bool {
    if (b.it_ == nullptr) {
      if (a.it_ == nullptr) {
        return false;
      }
      return *a.it_ != '\0';
    }
    return a.it_ != b.it_;
  }

  [[nodiscard]] constexpr friend auto operator==(
      const CStringIterator& a, const CStringIterator& b) noexcept -> bool {
    return !(a != b);
  }

  constexpr auto operator--() noexcept -> CStringIterator& {
    --it_;
    return *this;
  }

  constexpr auto operator--(int) noexcept -> CStringIterator {
    CStringIterator tmp(*this);
    --*this;
    return tmp;
  }

  constexpr auto operator+=(const difference_type offset) noexcept
      -> CStringIterator& {
    it_ += offset;
    return *this;
  }

  constexpr auto operator-=(const difference_type offset) noexcept
      -> CStringIterator& {
    it_ -= offset;
    return *this;
  }

  [[nodiscard]] constexpr auto operator+(
      const difference_type offset) const noexcept -> CStringIterator {
    CStringIterator tmp(*this);
    tmp += offset;
    return tmp;
  }

  [[nodiscard]] constexpr auto operator-(
      const difference_type offset) const noexcept -> CStringIterator {
    CStringIterator tmp(*this);
    tmp -= offset;
    return tmp;
  }

  [[nodiscard]] constexpr friend auto operator-(
      const CStringIterator& a, const CStringIterator& b) noexcept
      -> difference_type {
    DOMINO_LAZY_ASSERT(a.it_ != nullptr && b.it_ != nullptr,
                       "Incompatible iterator types");
    return a.it_ - b.it_;
  }

  [[nodiscard]] constexpr auto operator[](
      const difference_type offset) const noexcept -> value_type {
    return *(*this + offset);
  }

  [[nodiscard]] constexpr friend auto operator<(
      const CStringIterator& a, const CStringIterator& b) noexcept -> bool {
    DOMINO_LAZY_ASSERT(a.it_ != nullptr && b.it_ != nullptr,
                       "Incompatible iterator types");
    return a.it_ < b.it_;
  }

  [[nodiscard]] constexpr friend auto operator>(
      const CStringIterator& a, const CStringIterator& b) noexcept -> bool {
    return b < a;
  }

  [[nodiscard]] constexpr friend auto operator<=(
      const CStringIterator& a, const CStringIterator& b) noexcept -> bool {
    return !(b < a);  // NOLINT
  }

  [[nodiscard]] constexpr friend auto operator>=(
      const CStringIterator& a, const CStringIterator& b) noexcept -> bool {
    return !(a < b);  // NOLINT
  }

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return it_ != nullptr && *it_ != '\0';
  }

 private:
  const C* it_ = nullptr;
};

}  // namespace domino::lazy::internal

#endif  // DOMINO_LAZY_DETAIL_CSTRINGITERATOR_HPP_