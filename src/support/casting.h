#ifndef DOMINO_SUPPORT_CASTING_H_
#define DOMINO_SUPPORT_CASTING_H_

namespace domino {

template <typename To, typename From, typename Enable = void>
struct isa_impl {
  static inline bool doit(const From& Val) { return To::classof(&Val); }
};

}  // namespace domino

#endif  // DOMINO_SUPPORT_CASTING_H_