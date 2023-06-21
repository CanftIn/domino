#ifndef DOMINO_UTIL_ANY_H_
#define DOMINO_UTIL_ANY_H_

#include <domino/util/STLExtras.h>

#include <cassert>
#include <memory>
#include <type_traits>

namespace domino {

class any {
  template <typename T>
  struct TypeId {
    static char id;
  };

  template <typename T>
  struct IsInPlaceType;

  struct StorageBase {
    virtual ~StorageBase() = default;
    virtual auto clone() const -> std::unique_ptr<StorageBase> = 0;
    virtual auto id() const noexcept -> const void* = 0;
  };

  template <typename T>
  struct StorageImpl : public StorageBase {
    template <typename... Args>
    explicit StorageImpl(std::in_place_t /*unused*/, Args&&... args)
        : value(std::forward<Args>(args)...) {}

    auto clone() const -> std::unique_ptr<StorageBase> final {
      return std::unique_ptr<StorageBase>(
          new StorageImpl(std::in_place, value));
    }

    auto id() const noexcept -> const void* final { return &TypeId<T>::id; }

    T value;

    auto operator=(const StorageImpl& other) -> StorageImpl& = delete;
    StorageImpl(const StorageImpl& other) = delete;
  };

 public:
  constexpr any() noexcept = default;

  any(const any& other)
      : obj_(other.has_value() ? other.obj_->clone()
                               : std::unique_ptr<StorageBase>()) {}

  template <typename T, typename VT = std::decay_t<T>,
            std::enable_if_t<
                std::conjunction_v<std::negation<std::is_same<VT, any>>,
                                   std::negation<IsInPlaceType<VT>>,
                                   std::negation<std::is_convertible<any, VT>>,
                                   std::is_copy_constructible<VT>>,
                int> = 0>
  any(T&& value)
      : obj_(new StorageImpl<VT>(std::in_place, std::forward<T>(value))) {}

  template <
      typename T, typename... Args, typename VT = std::decay_t<T>,
      std::enable_if_t<std::conjunction_v<std::is_copy_constructible<VT>,
                                          std::is_constructible<VT, Args...>>,
                       int> = 0>
  explicit any(std::in_place_type_t<T> /*unused*/, Args&&... args)
      : obj_(new StorageImpl<VT>(std::in_place, std::forward<Args>(args)...)) {}

  template <
      typename T, typename U, typename... Args, typename VT = std::decay_t<T>,
      std::enable_if_t<
          std::conjunction_v<
              std::is_copy_constructible<VT>,
              std::is_constructible<VT, std::initializer_list<U>&, Args...>>,
          int> = 0>
  explicit any(std::in_place_type_t<T> /*unused*/,
               std::initializer_list<U> ilist, Args&&... args)
      : obj_(new StorageImpl<VT>(std::in_place, ilist,
                                 std::forward<Args>(args)...)) {}

  any(any&& other) noexcept = default;

  auto operator=(const any& other) -> any& {
    any(other).swap(*this);
    return *this;
  }

  auto operator=(any&& other) noexcept -> any& {
    any(std::move(other)).swap(*this);
    return *this;
  }

  template <typename T, typename VT = std::decay_t<T>,
            std::enable_if_t<
                std::conjunction_v<std::negation<std::is_same<VT, any>>,
                                   std::negation<IsInPlaceType<VT>>,
                                   std::negation<std::is_convertible<any, VT>>,
                                   std::is_copy_constructible<VT>>,
                int> = 0>
  auto operator=(T&& value) -> any& {
    any(std::in_place_type_t<VT>(), std::forward<T>(value)).swap(*this);
    return *this;
  }

  template <typename T, typename... Args, typename VT = std::decay_t<T>,
            std::enable_if_t<std::is_copy_constructible_v<VT> &&
                                 std::is_constructible_v<VT, Args...>,
                             int> = 0>
  auto emplace(Args&&... args) -> VT& {
    reset();
    auto* const object_ptr =
        new StorageImpl<VT>(std::in_place, std::forward<Args>(args)...);
    obj_ = std::unique_ptr<StorageBase>(object_ptr);
    return object_ptr->value;
  }

  template <
      typename T, typename U, typename... Args, typename VT = std::decay_t<T>,
      std::enable_if_t<
          std::is_copy_constructible_v<VT> &&
              std::is_constructible_v<VT, std::initializer_list<U>&, Args...>,
          int> = 0>
  auto emplace(std::initializer_list<U> ilist, Args&&... args) -> VT& {
    reset();
    auto* const object_ptr =
        new StorageImpl<VT>(std::in_place, ilist, std::forward<Args>(args)...);
    obj_ = std::unique_ptr<StorageBase>(object_ptr);
    return object_ptr->value;
  }

  auto has_value() const noexcept -> bool { return obj_ != nullptr; }

  auto reset() noexcept -> void { obj_.reset(); }

  auto swap(any& other) noexcept -> void { obj_.swap(other.obj_); }

 private:
  template <typename T>
  auto isa() const -> bool {
    if (!obj_) {
      return false;
    }
    return obj_->id() == &any::TypeId<remove_cvref_t<T>>::id;
  }

  auto CloneObj() const -> std::unique_ptr<StorageBase> {
    if (!obj_) {
      return nullptr;
    }
    return obj_->clone();
  }

  template <class T>
  friend auto any_cast(const any& value) -> T;
  template <class T>
  friend auto any_cast(any& value) -> T;
  template <class T>
  friend auto any_cast(any&& value) -> T;
  template <class T>
  friend auto any_cast(const any* value) noexcept -> const T*;
  template <class T>
  friend auto any_cast(any* value) noexcept -> T*;
  template <typename T>
  friend auto any_isa(const any& value) -> bool;

  std::unique_ptr<StorageBase> obj_;
};

template <typename T>
char any::TypeId<T>::id = 0;

template <typename T>
struct any::IsInPlaceType : std::false_type {};

template <typename T>
struct any::IsInPlaceType<std::in_place_type_t<T>> : std::true_type {};

inline void swap(any& x, any& y) noexcept { x.swap(y); }

template <typename T, typename... Args>
auto make_any(Args&&... args) -> any {
  return any(std::in_place_type_t<T>(), std::forward<Args>(args)...);
}

template <typename T, typename U, typename... Args>
auto make_any(std::initializer_list<U> il, Args&&... args) -> any {
  return any(std::in_place_type_t<T>(), il, std::forward<Args>(args)...);
}

template <typename T>
auto any_isa(const any& value) -> bool {
  return value.isa<T>();
}

template <class T>
auto any_cast(const any& value) -> T {
  assert(value.isa<T>() && "Bad any cast!");
  return static_cast<T>(*any_cast<remove_cvref_t<T>>(&value));
}

template <class T>
auto any_cast(any& value) -> T {
  assert(value.isa<T>() && "Bad any cast!");
  return static_cast<T>(*any_cast<remove_cvref_t<T>>(&value));
}

template <class T>
auto any_cast(any&& value) -> T {
  assert(value.isa<T>() && "Bad any cast!");
  return static_cast<T>(std::move(*any_cast<remove_cvref_t<T>>(&value)));
}

template <class T>
auto any_cast(const any* value) noexcept -> const T* {
  using U = remove_cvref_t<T>;
  if (!value || !value->isa<U>()) {
    return nullptr;
  }
  return &static_cast<any::StorageImpl<U>&>(*value->obj_).value;
}

template <class T>
auto any_cast(any* value) noexcept -> T* {
  using U = std::decay_t<T>;
  if (!value || !value->isa<U>()) {
    return nullptr;
  }
  return &static_cast<any::StorageImpl<U>&>(*value->obj_).value;
}

}  // namespace domino

#endif  // DOMINO_UTIL_ANY_H_