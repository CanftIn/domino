#ifndef DOMINO_RPC_TRANSPORT_VIRTUALTRANSPORT_H_
#define DOMINO_RPC_TRANSPORT_VIRTUALTRANSPORT_H_

#include <domino/rpc/Configuration.h>
#include <domino/rpc/transport/Transport.h>

#include <memory>

namespace domino {
namespace rpc {
namespace transport {

class TransportDefaults : public Transport {
 public:
  uint32_t read(uint8_t* buf, uint32_t len) {
    return this->read_virt(buf, len);
  }

  void write(const uint8_t* buf, uint32_t len) { this->write_virt(buf, len); }

  void consume(uint32_t len) { this->consume_virt(len); }

 protected:
  TransportDefaults(std::shared_ptr<Configuration> config = nullptr)
      : Transport(config) {}
};

template <class TransportType, class Super = TransportDefaults>
class VirtualTransport : public Super {
 public:
  uint32_t read_virt(uint8_t* buf, uint32_t len) override {
    return static_cast<TransportType*>(this)->read(buf, len);
  }

  void write_virt(const uint8_t* buf, uint32_t len) override {
    static_cast<TransportType*>(this)->write(buf, len);
  }

  void consume_virt(uint32_t len) override {
    static_cast<TransportType*>(this)->consume(len);
  }

 protected:
  VirtualTransport() : Super() {}

  template <typename Arg>
  VirtualTransport(const Arg& arg) : Super(arg) {}

  template <typename Arg1, typename Arg2>
  VirtualTransport(const Arg1& arg1, const Arg2& arg2) : Super(arg1, arg2) {}
};

}  // namespace transport
}  // namespace rpc
}  // namespace domino

#endif  // DOMINO_RPC_TRANSPORT_VIRTUALTRANSPORT_H_