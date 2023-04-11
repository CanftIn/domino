#include <domino/util/logging.h>
#include <gtest/gtest.h>

TEST(Logging, debug) {
  std::string str = "[this is a file]";
  DOMINO_DEBUG("%s", "Trying to double-init TFileTransport");
  DOMINO_DEBUG(
      "TFileTransport: unable to reopen log file %s during error recovery",
      str.c_str());
}

TEST(Logging, debug_t) {
  std::string str = "[this is a file]";
  DOMINO_DEBUG_T("%s", "Trying to double-init TFileTransport");
  DOMINO_DEBUG_T(
      "TFileTransport: unable to reopen log file %s during error recovery",
      str.c_str());
}

TEST(Logging, error) {
  std::string str = "[this is a file]";
  DOMINO_ERROR("%s", "Trying to double-init TFileTransport");
  DOMINO_ERROR(
      "TFileTransport: unable to reopen log file %s during error recovery",
      str.c_str());
}

TEST(Logging, log_oper) {
  std::string str = "[this is a file]";
  DOMINO_LOG_OPER("%s", "Trying to double-init TFileTransport");
  DOMINO_LOG_OPER(
      "TFileTransport: unable to reopen log file %s during error recovery",
      str.c_str());
}