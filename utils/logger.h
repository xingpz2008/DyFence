#pragma once

#include <iostream>
#include "../config.h"

// Convenience macros for colored terminal output.

#define GREEN(STRING)  std::cout<<"\033[32m"<<STRING<<"\033[m\n"
#define RED(STRING)    std::cout<<"\033[31m"<<STRING<<"\033[m\n"
#define YELLOW(STRING) std::cout<<"\033[33m"<<STRING<<"\033[m\n"

#define BLUE(STRING)    std::cout<<"\033[34m"<<STRING<<"\033[m\n"
#define MAGENTA(STRING) std::cout<<"\033[35m"<<STRING<<"\033[m\n"
#define CYAN(STRING)    std::cout<<"\033[36m"<<STRING<<"\033[m\n"
#define WHITE(STRING)   std::cout<<"\033[37m"<<STRING<<"\033[m\n"

#define YELLOW_WITHOUTENTER(STRING) std::cout<<"\033[33m"<<STRING<<"\033[m"
#define GREEN_WITHOUTENTER(STRING)  std::cout<<"\033[32m"<<STRING<<"\033[m"
#define RED_WITHOUTENTER(STRING)    std::cout<<"\033[31m"<<STRING<<"\033[m"

#define BLUE_WITHOUTENTER(STRING)    std::cout<<"\033[34m"<<STRING<<"\033[m"
#define MAGENTA_WITHOUTENTER(STRING) std::cout<<"\033[35m"<<STRING<<"\033[m"
#define CYAN_WITHOUTENTER(STRING)    std::cout<<"\033[36m"<<STRING<<"\033[m"
#define WHITE_WITHOUTENTER(STRING)   std::cout<<"\033[37m"<<STRING<<"\033[m"

// ANSI color escape sequences used by the logging macros.
#define COLOR_GREEN  "\033[32m"
#define COLOR_RED    "\033[31m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_NORMAL "\033[m"

#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_WHITE   "\033[37m"

// Runtime logging threshold. Lower levels include more messages.
enum LogLevel {
    LEVEL_DEBUG = 0,
    LEVEL_INFO  = 1,
    LEVEL_WARN  = 2,
    LEVEL_ERROR = 3,
    LEVEL_NONE  = 4
};

inline LogLevel& GlobalLogLevel() {
    static LogLevel level = MVZK_CONFIG_DEFAULT_LEVEL_INFO;
    return level;
}

inline void SetLogLevel(LogLevel level) {
    GlobalLogLevel() = level;
}

// Stream-style logging macros.
#define LOG_DEBUG(MSG) \
    do { if (GlobalLogLevel() <= LEVEL_DEBUG) { \
        std::cout << COLOR_CYAN << "[DEBUG] " << MSG << COLOR_NORMAL << "\n"; \
    } } while(0)

#define LOG_INFO(MSG) \
    do { if (GlobalLogLevel() <= LEVEL_INFO) { \
        std::cout << COLOR_GREEN << "[INFO]  " << MSG << COLOR_NORMAL << "\n"; \
    } } while(0)

#define LOG_WARN(MSG) \
    do { if (GlobalLogLevel() <= LEVEL_WARN) { \
        std::cout << COLOR_YELLOW << "[WARN]  " << MSG << COLOR_NORMAL << "\n"; \
    } } while(0)

#define LOG_ERROR(MSG) \
    do { if (GlobalLogLevel() <= LEVEL_ERROR) { \
        std::cerr << COLOR_RED << "[ERROR] " << MSG << COLOR_NORMAL << "\n"; \
    } } while(0)
