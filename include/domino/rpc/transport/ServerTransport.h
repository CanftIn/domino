#ifndef DOMINO_RPC_TRANSPORT_SERVERTRANSPORT_H_
#define DOMINO_RPC_TRANSPORT_SERVERTRANSPORT_H_

#include <domino/rpc/transport/SocketDefine.h>
#include <domino/rpc/transport/Transport.h>

#include <memory>

namespace domino {
namespace rpc {
namespace transport {

class ServerTransport {
 public:
  virtual ~ServerTransport() = default;

  virtual bool isOpen() const { return false; }

  virtual void listen() = 0;

  virtual void interrupt() = 0;

  virtual void close() = 0;

  virtual DOMINO_SOCKET getSocketFD() { return -1; }

  std::shared_ptr<Transport> accept() {
    std::shared_ptr<Transport> result = acceptImpl();
    if (!result) {
      // TODO: Exception handle
    }
    return result;
  }

 protected:
  ServerTransport() = default;

  virtual std::shared_ptr<Transport> acceptImpl() = 0;
};

}  // namespace transport
}  // namespace rpc
}  // namespace domino

#endif  // DOMINO_RPC_TRANSPORT_SERVERTRANSPORT_H_