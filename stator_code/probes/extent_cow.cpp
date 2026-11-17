#include "extent_cow.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include <sys/mman.h>

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

namespace hostmm {
namespace {

double elapsed_ms(const std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

template <typename Task>
void parallel(const unsigned threads, const Task& task) {
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (unsigned index = 1U; index < threads; ++index) {
    workers.emplace_back(task, index);
  }
  task(0U);
  for (std::thread& worker : workers) {
    worker.join();
  }
}

}

bool privatize_range(void* const base, const std::size_t bytes,
                     const std::size_t extent_bytes, const unsigned threads,
                     const bool huge_pages, PrivatizeStats* const stats) {
  if (reinterpret_cast<std::uintptr_t>(base) % kHugeBytes != 0U ||
      extent_bytes == 0U || extent_bytes % kHugeBytes != 0U || threads == 0U) {
    errno = EINVAL;
    return false;
  }
  PrivatizeStats local;
  for (std::size_t offset = 0U; offset < bytes; offset += extent_bytes) {
    const std::size_t length = std::min(extent_bytes, bytes - offset);
    char* const destination = static_cast<char*>(base) + offset;

    const auto copy_start = std::chrono::steady_clock::now();
    const std::size_t mapped = length + kHugeBytes;
    void* const raw = ::mmap(nullptr, mapped, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
      return false;
    }
    char* const staging = reinterpret_cast<char*>(
        (reinterpret_cast<std::uintptr_t>(raw) + kHugeBytes - 1U) &
        ~(kHugeBytes - 1U));
    const std::size_t head = static_cast<std::size_t>(staging -
                                                      static_cast<char*>(raw));
    if (head != 0U) {
      (void)::munmap(raw, head);
    }
    const std::size_t tail = mapped - head - length;
    if (tail != 0U) {
      (void)::munmap(staging + length, tail);
    }
    (void)::madvise(staging, length,
                    huge_pages ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);

    const unsigned workers = static_cast<unsigned>(
        std::min<std::size_t>(threads, std::max<std::size_t>(
                                           1U, length / kHugeBytes)));
    const std::size_t slice =
        ((length / workers) + kHugeBytes - 1U) & ~(kHugeBytes - 1U);
    parallel(workers, [&](const unsigned index) {
      const std::size_t begin = std::min(length, index * slice);
      const std::size_t end = std::min(length, begin + slice);
      if (begin < end) {
        std::memcpy(staging + begin, destination + begin, end - begin);
      }
    });
    local.copy_ms += elapsed_ms(copy_start);

    const auto remap_start = std::chrono::steady_clock::now();
    if (::mremap(staging, length, length, MREMAP_MAYMOVE | MREMAP_FIXED,
                 destination) == MAP_FAILED) {
      const int saved = errno;
      (void)::munmap(staging, length);
      errno = saved;
      return false;
    }
    local.remap_ms += elapsed_ms(remap_start);
    ++local.extents;
  }
  if (stats != nullptr) {
    *stats = local;
  }
  return true;
}

double populate_write(void* const base, const std::size_t bytes,
                      const unsigned threads) {
  if (threads == 0U) {
    return -1.0;
  }
  const auto start = std::chrono::steady_clock::now();
  const std::size_t slice =
      ((bytes / threads) + kHugeBytes - 1U) & ~(kHugeBytes - 1U);
  std::atomic<bool> failed{false};
  parallel(threads, [&](const unsigned index) {
    const std::size_t begin = std::min(bytes, index * slice);
    const std::size_t end = std::min(bytes, begin + slice);
    if (begin < end &&
        ::madvise(static_cast<char*>(base) + begin, end - begin,
                  MADV_POPULATE_WRITE) != 0) {
      failed.store(true);
    }
  });
  return failed.load() ? -1.0 : elapsed_ms(start);
}

}
