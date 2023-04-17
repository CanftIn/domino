#ifndef DOMINO_RPC_TRANSPORT_TRANSPORT_H_
#define DOMINO_RPC_TRANSPORT_TRANSPORT_H_

#include <domino/rpc/Configuration.h>
#include <domino/util/StringRef.h>

#include <cstdint>
#include <memory>

namespace domino {
namespace rpc {
namespace transport {

template <class Transport>
uint32_t readAll(Transport& trans, uint8_t* buf, uint32_t len) {
  uint32_t have = 0;
  uint32_t get = 0;

  while (have < len) {
    get = trans.read(buf + have, len - have);
    if (get <= 0) {
      // TODO: Exception handle
    }
    have += get;
  }
  return have;
}

class Transport {
 public:
  Transport(std::shared_ptr<Configuration> config = nullptr) {
    if (config == nullptr) {
      configuration_ = std::shared_ptr<Configuration>(new Configuration());
    } else {
      configuration_ = config;
    }
    resetConsumeMessageSize();
  }
  virtual ~Transport() = default;

  virtual bool isOpen() const { return false; }

  virtual bool peek() { return isOpen(); }

  virtual void open() {}

  virtual void close() {}

  uint32_t read(uint8_t* buf, uint32_t len) {
    uint32_t got = read_virt(buf, len);
    consume(got);
    return got;
  }

  virtual uint32_t read_virt(uint8_t* buf, uint32_t len) { return 0; }

  virtual uint32_t readEnd() { return 0; }

  void write(const uint8_t* buf, uint32_t len) {
    write_virt(buf, len);
    consume(len);
  }

  virtual void write_virt(const uint8_t* buf, uint32_t len) {}

  virtual uint32_t writeEnd() { return 0; }

  virtual void flush() {}

  void consume(uint32_t len) { consume_virt(len) }

  virtual void consume_virt(uint32_t len) {
    // TODO: function body
  }

  virtual const StringRef getOrigin() const { return "Unknown"; }

  std::shared_ptr<Configuration> getConfiguration() { return configuration_; }

  void setConfiguration(std::shared_ptr<Configuration> config) {
    if (config != nullptr) configuration_ = config;
  }

  void resetConsumeMessageSize(long newSize = -1) {}

 protected:
  std::shared_ptr<Configuration> configuration_;
};

class TransportFactory {
 public:
  TransportFactory() = default;

  virtual ~TransportFactory() = default;

  virtual std::shared_ptr<Transport> getTransport(
      std::shared_ptr<Transport> trans) {
    return trans;
  }
};

}  // namespace transport
}  // namespace rpc
}  // namespace domino

#endif  // DOMINO_RPC_TRANSPORT_TRANSPORT_H_