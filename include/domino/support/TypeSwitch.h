#ifndef DOMINO_SUPPORT_TYPESWITCH_H_
#define DOMINO_SUPPORT_TYPESWITCH_H_

#include <domino/util/STLExtras.h>
#include <domino/support/Casting.h>

#include <optional>

namespace domino {

namespace detail {

template <typename DerivedT, typename T>
class TypeSwitchBase {
 public:
  TypeSwitchBase(const T& value) : value(value) {}
  TypeSwitchBase(TypeSwitchBase&& other) : value(other.value) {}
  ~TypeSwitchBase() = default;

  TypeSwitchBase(const TypeSwitchBase&) = delete;
  void operator=(const TypeSwitchBase&) = delete;
  void operator=(TypeSwitchBase&& other) = delete;
  
 protected:
  const T value;
};

}

}

#endif  // DOMINO_SUPPORT_TYPESWITCH_H_