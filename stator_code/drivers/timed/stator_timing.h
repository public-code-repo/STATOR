#ifndef STATOR_TIMING_H
#define STATOR_TIMING_H

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <time.h>
#include <unistd.h>

namespace stator_measure {
inline int64_t monotonic_ns() {
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    throw std::system_error(errno, std::generic_category(), "CLOCK_MONOTONIC");
  }
  return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

class Recorder {
 public:
  Recorder() : pid_(getpid()) {
    const char* directory = std::getenv("STATOR_TIMING_DIR");
    const char* enabled = std::getenv("STATOR_TRACE");
    enabled_ = directory && *directory && !(enabled && std::string(enabled) == "0");
    if (!enabled_) return;
    const int64_t start = monotonic_ns();
    path_ = std::string(directory) + "/" + std::to_string(pid_) + "-" +
            std::to_string(start) + ".jsonl";
    events_.reserve(512);
    events_.push_back({"process_start", start, 0, 0, 0});
  }
  Recorder(const Recorder&) = delete;
  Recorder& operator=(const Recorder&) = delete;
  ~Recorder() noexcept {
    if (finished_ || !enabled_) return;
    try {
      event(std::uncaught_exceptions() ? "exception" : "process_end");
      checkpoint();
    } catch (const std::exception& error) {
      std::fprintf(stderr, "STATOR_TIMING_ERROR %s\n", error.what());
    }
    if (file_) std::fclose(file_);
  }
  void reserve(size_t count) {
    if (enabled_) events_.reserve(count);
  }
  void event(const char* name, int64_t value = 0, int sequence = 0) {
    if (enabled_) events_.push_back({name, monotonic_ns(), generation_, value, sequence});
  }
  void begin_generation() {
    ++generation_;
    event("generation_start");
  }
  void token(int64_t id, int sequence = 0) { event("token", id, sequence); }
  void end_generation(int64_t count) { event("generation_end", count); }
  void checkpoint() {
    if (!enabled_) return;
    if (!file_) {
      file_ = std::fopen(path_.c_str(), "wx");
      if (!file_) throw std::system_error(errno, std::generic_category(), "timing open");
    }
    for (; written_ < events_.size(); ++written_) {
      const auto& e = events_[written_];
      if (std::fprintf(file_, "{\"schema\":1,\"pid\":%d,\"event\":\"%s\","
                       "\"mono_ns\":%lld,\"generation\":%d,\"value\":%lld,"
                       "\"sequence\":%d}\n", int(pid_), e.name,
                       static_cast<long long>(e.time), e.generation,
                       static_cast<long long>(e.value), e.sequence) < 0) {
        throw std::system_error(errno ? errno : EIO, std::generic_category(), "timing write");
      }
    }
    if (std::fflush(file_) != 0) {
      throw std::system_error(errno, std::generic_category(), "timing flush");
    }
  }
  void finish() {
    if (!enabled_ || finished_) return;
    event("process_end");
    checkpoint();
    FILE* closing = file_;
    file_ = nullptr;
    finished_ = true;
    if (std::fclose(closing) != 0) {
      throw std::system_error(errno, std::generic_category(), "timing close");
    }
  }
 private:
  struct Event { const char* name; int64_t time; int generation; int64_t value; int sequence; };
  pid_t pid_;
  bool enabled_ = false;
  bool finished_ = false;
  int generation_ = 0;
  size_t written_ = 0;
  std::string path_;
  std::vector<Event> events_;
  FILE* file_ = nullptr;
};
}
#endif
