#ifndef DOMINO_RPC_TRANSPORT_SERVERSOCKET_H_
#define DOMINO_RPC_TRANSPORT_SERVERSOCKET_H_

#include <domino/rpc/concurrency/Mutex.h>
#include <domino/rpc/transport/ServerTransport.h>
#include <domino/rpc/transport/SocketDefine.h>
#include <sys/types.h>

#include <functional>

namespace domino {
namespace rpc {
namespace transport {

class ServerSocket : public ServerTransport {
 public:
  using socket_func_type = std::function<void(DOMINO_SOCKET fd)>;
  
  static const int DEFAULT_BACKLOG = 1024;

  ServerSocket(int port);

  ServerSocket(int port, int sendTimeout, int recvTiemout);

  ServerSocket(const std::string& address, int port);

  ServerSocket(const std::string& path);

  ~ServerSocket() override;

  void setSendTimeout(int sendTimeout);

  void setRecvTimeout(int recvTimeout);

  void setAcceptTimeout(int accTimeout);

  void setAcceptBacklog(int accBacklog);

  void setRetryLimit(int retryLimit);

  void setRetryDelay(int retryDelay);

  void setKeepAlive(bool keepAlive) { keepAlive_ = keepAlive; }

  bool isOpen() const override;

  void listen() override;

  void interrupt() override;

  void close() override;

 private:
  int port_;
  std::string address_;
  std::string path_;
  DOMINO_SOCKET serverSocket_;
  int acceptBacklog_;
  int sendTimeout_;
  int recvTimeout_;
  int accTimeout_;
  int retryLimit_;
  int retryDelay_;
  int tcpSendBuffer_;
  int tcpRecvBuffer_;
  bool keepAlive_;
  bool listening_;

  concurrency::Mutex rwMutex_;
  DOMINO_SOCKET interruptSockWriter_;
  DOMINO_SOCKET interruptSockReader_;
  
  socket_func_type listenCallback_;
  socket_func_type acceptCallback_;
};

}  // namespace transport
}  // namespace rpc
}  // namespace domino

#endif  // DOMINO_RPC_TRANSPORT_SERVERSOCKET_H_