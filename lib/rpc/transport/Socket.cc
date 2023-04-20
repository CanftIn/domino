#include <domino/rpc/transport/Socket.h>
#include <fcntl.h>
#include <sys/types.h>

#include <cstring>
#include <sstream>

template <class T>
inline const void* const_cast_sockopt(const T* V) {
  return reinterpret_cast<const void*>(V);
}

namespace domino {

namespace rpc {

namespace transport {


  
}

}  // namespace rpc

}  // namespace domino