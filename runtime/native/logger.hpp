#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace logger {

enum class LogLevel { kInfo, kWarn, kError, kDebug };

inline const char* levelToStr(LogLevel level) {
    switch (level) {
        case LogLevel::kInfo: return "INFO";
        case LogLevel::kWarn: return "WARN";
        case LogLevel::kError: return "ERROR";
        case LogLevel::kDebug: return "DEBUG";
        default: return "INFO";
    }
}

inline std::string nowString() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm{};
    localtime_r(&t, &tm);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%F %T") << '.' << std::setw(3) << std::setfill('0') << ms.count();
    return oss.str();
}

inline void log(LogLevel level, const std::string& tag, const std::string& msg) {
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    std::cout << '[' << nowString() << "] [" << levelToStr(level) << "] [" << tag << "] " << msg << std::endl;
}

}  // namespace logger

#define LOG_INFO(tag, msg) ::logger::log(::logger::LogLevel::kInfo, tag, msg)
#define LOG_WARN(tag, msg) ::logger::log(::logger::LogLevel::kWarn, tag, msg)
#define LOG_ERROR(tag, msg) ::logger::log(::logger::LogLevel::kError, tag, msg)
#define LOG_DEBUG(tag, msg) ::logger::log(::logger::LogLevel::kDebug, tag, msg)

#endif // LOGGER_HPP
