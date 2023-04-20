#include <asm-generic/ioctls.h>
#include <domino/rpc/transport/Socket.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/types.h>

#include <cstring>
#include <sstream>

template <class T>
inline void* cast_sockopt(const T* V) {
  return reinterpret_cast<void*>(V);
}

template <class T>
inline const void* const_cast_sockopt(const T* V) {
  return reinterpret_cast<const void*>(V);
}

namespace domino {

namespace rpc {

namespace transport {

Socket::Socket(std::shared_ptr<Configuration> config)
    : VirtualTransport(config),
      port_(0),
      socket_(DOMINO_INVALID_SOCKET),
      peerPort_(0),
      connTimeout_(0),
      sendTimeout_(0),
      recvTimeout_(0),
      keepAlive_(false),
      noDelay_(true),
      maxRecvRetries_(5) {
  cachedPeerAddr_.ipv4.sin_family = AF_UNSPEC;
}

Socket::Socket(const std::string& host, uint16_t port,
               std::shared_ptr<Configuration> config)
    : VirtualTransport(config),
      host_(host),
      port_(port),
      socket_(DOMINO_INVALID_SOCKET),
      peerPort_(0),
      connTimeout_(0),
      sendTimeout_(0),
      recvTimeout_(0),
      keepAlive_(false),
      noDelay_(true),
      maxRecvRetries_(5) {
  if (host_.empty()) {
    throw std::invalid_argument("host cannot be empty");
  }
}

Socket::Socket(DOMINO_SOCKET socket, std::shared_ptr<Configuration> config)
    : VirtualTransport(config),
      port_(0),
      socket_(socket),
      peerPort_(0),
      connTimeout_(0),
      sendTimeout_(0),
      recvTimeout_(0),
      keepAlive_(false),
      noDelay_(true),
      maxRecvRetries_(5) {
  cachedPeerAddr_.ipv4.sin_family = AF_UNSPEC;
}

Socket::Socket(DOMINO_SOCKET socket,
               std::shared_ptr<DOMINO_SOCKET> interruptSocket,
               std::shared_ptr<Configuration> config)
    : VirtualTransport(config),
      socket_(socket),
      interruptListener_(interruptSocket),
      peerPort_(0),
      connTimeout_(0),
      sendTimeout_(0),
      recvTimeout_(0),
      keepAlive_(false),
      noDelay_(true),
      maxRecvRetries_(5) {
  cachedPeerAddr_.ipv4.sin_family = AF_UNSPEC;
}

Socket::~Socket() { close(); }

bool Socket::isOpen() const { return socket_ != DOMINO_INVALID_SOCKET; }

bool Socket::hasPendingDataToRead() {
  if (!isOpen()) return false;

  int32_t retries = 0;
  int numBytesAvailable;
try_again:
  int ret = ::ioctl(socket_, FIONREAD, &numBytesAvailable);
  if (ret == -1) {
    if (errno == EINTR &&(++retries < maxRecvRetries_)) {
      goto try_again;
    }
    // TODO: Error handling
  }
  return numBytesAvailable > 0;
}

}  // namespace transport

}  // namespace rpc

}  // namespace domino