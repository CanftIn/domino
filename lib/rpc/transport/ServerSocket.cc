#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <cstring>
#include <memory>
#include <stdexcept>

#include <domino/rpc/transport/ServerSocket.h>

#ifndef AF_LOCAL
#define AF_LOCAL AF_UNIX
#endif

