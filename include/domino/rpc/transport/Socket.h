#ifndef DOMINO_RPC_TRANSPORT_SOCKET_H_
#define DOMINO_RPC_TRANSPORT_SOCKET_H_

#include <domino/rpc/transport/SocketDefine.h>
#include <domino/rpc/transport/VirtualTransport.h>

namespace domino {
namespace rpc {
namespace transport {

class Socket : public VirtualTransport<Socket> {
 public:
  Socket(std::shared_ptr<Configuration> config = nullptr);

  Socket(const std::string& host, uint16_t port,
         std::shared_ptr<Configuration> config = nullptr);

  Socket(DOMINO_SOCKET socket, std::shared_ptr<Configuration> config = nullptr);

  Socket(DOMINO_SOCKET socket, std::shared_ptr<DOMINO_SOCKET> interruptSocket,
         std::shared_ptr<Configuration> config = nullptr);

  ~Socket() override;

  bool isOpen() const override;

  bool peek() override;

  void open() override;

  void close() override;

  virtual bool hasPendingDataToRead();

  virtual uint32_t read(uint8_t* buf, uint32_t len);

  virtual void write(const uint8_t* buf, uint32_t len);

  virtual uint32_t write_partial(const uint8_t* buf, uint32_t len);

  std::string getHost() const;

  uint16_t getPort() const;

  bool isUnixDomainSocket() const;

  void setHost(std::string host);

  void setPort(uint16_t port);

  void setNoDelay(bool noDelay);

  void setConnTimeout(uint32_t ms);

  void setRecvTimeout(uint32_t ms);

  void setSendTimeout(uint32_t ms);

  void setMaxRecvRetries(uint32_t MaxRecvRetries);

  void setKeepAlive(bool keepAlive);

  std::string getSocketInfo() const;

  std::string getPeerHost() const;

  std::string getPeerAddress() const;

  uint16_t getPeerPort() const;

  DOMINO_SOCKET getSocketFD() override { return socket_; }

  void setSocketFD(DOMINO_SOCKET fd) { socket_ = fd; }

  sockaddr* getCachedAddress(socklen_t* len) const;

 protected:
  void openConnection(struct addrinfo* addr);

  std::string host_;
  uint16_t port_;
  DOMINO_SOCKET socket_;
  mutable std::string peerHost_;
  mutable std::string peerAddress_;
  mutable uint16_t peerPort_;
  std::shared_ptr<DOMINO_SOCKET> interruptListener_;
  int connTimeout_;
  int recvTimeout_;
  int sendTimeout_;
  bool keepAlive_;
  bool noDelay_;
  uint32_t maxRecvRetries_;
  union {
    struct sockaddr_in addr4_;
    struct sockaddr_in6 addr6_;
  } cachedPeerAddr_;

 private:
  void unix_open();
  void local_open();
};

}  // namespace transport
}  // namespace rpc
}  // namespace domino

#endif  // DOMINO_RPC_TRANSPORT_SOCKET_H_