#ifndef DOMINO_UTIL_LOGGING_H_
#define DOMINO_UTIL_LOGGING_H_

#include <time.h>

/**
 * GLOBAL_DEBUGGING_LEVEL = 0: all debugging turned off, debug macros undefined
 * GLOBAL_DEBUGGING_LEVEL = 1: all debugging turned on
 */
#define GLOBAL_DEBUGGING_LEVEL 1

/**
 * GLOBAL_LOGGING_LEVEL = 0: all logging turned off, logging macros undefined
 * GLOBAL_LOGGING_LEVEL = 1: all logging turned on
 */
#define GLOBAL_LOGGING_LEVEL 1

#if GLOBAL_DEBUGGING_LEVEL > 0
#define DOMINO_DEBUG(format_string, ...)                                \
  if (GLOBAL_DEBUGGING_LEVEL > 0) {                                     \
    fprintf(stderr, "[%s,%d] " format_string " \n", __FILE__, __LINE__, \
            ##__VA_ARGS__);                                             \
  }
#else
#define DOMINO_DEBUG(format_string, ...)
#endif

#if GLOBAL_DEBUGGING_LEVEL > 0
#define DOMINO_DEBUG_T(format_string, ...)                                     \
  {                                                                            \
    if (GLOBAL_DEBUGGING_LEVEL > 0) {                                          \
      time_t now;                                                              \
      char dbgtime[26];                                                        \
      time(&now);                                                              \
      ctime_r(&now, dbgtime);                                                  \
      dbgtime[24] = '\0';                                                      \
      fprintf(stderr, "[%s,%d] [%s] " format_string " \n", __FILE__, __LINE__, \
              dbgtime, ##__VA_ARGS__);                                         \
    }                                                                          \
  }
#else
#define DOMINO_DEBUG_T(format_string, ...)
#endif

#define DOMINO_DEBUG_L(level, format_string, ...)                       \
  if ((level) > 0) {                                                    \
    fprintf(stderr, "[%s,%d] " format_string " \n", __FILE__, __LINE__, \
            ##__VA_ARGS__);                                             \
  }

#define DOMINO_ERROR(format_string, ...)                                  \
  {                                                                       \
    time_t now;                                                           \
    char dbgtime[26];                                                     \
    time(&now);                                                           \
    ctime_r(&now, dbgtime);                                               \
    dbgtime[24] = '\0';                                                   \
    fprintf(stderr, "[%s,%d] [%s] ERROR: " format_string " \n", __FILE__, \
            __LINE__, dbgtime, ##__VA_ARGS__);                            \
  }

#define DOMINO_ERROR_ABORT(format_string, ...)                                 \
  {                                                                            \
    time_t now;                                                                \
    char dbgtime[26];                                                          \
    time(&now);                                                                \
    ctime_r(&now, dbgtime);                                                    \
    dbgtime[24] = '\0';                                                        \
    fprintf(stderr, "[%s,%d] [%s] ERROR: Going to abort " format_string " \n", \
            __FILE__, __LINE__, dbgtime, ##__VA_ARGS__);                       \
    exit(1);                                                                   \
  }

#if GLOBAL_LOGGING_LEVEL > 0
#define DOMINO_LOG_OPER(format_string, ...)                                 \
  {                                                                         \
    if (GLOBAL_LOGGING_LEVEL > 0) {                                         \
      time_t now;                                                           \
      char dbgtime[26];                                                     \
      time(&now);                                                           \
      ctime_r(&now, dbgtime);                                               \
      dbgtime[24] = '\0';                                                   \
      fprintf(stderr, "[%s] " format_string " \n", dbgtime, ##__VA_ARGS__); \
    }                                                                       \
  }
#else
#define DOMINO_LOG_OPER(format_string, ...)
#endif

#endif  // DOMINO_UTIL_LOGGING_H_