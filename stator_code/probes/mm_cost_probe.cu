#include <cuda_runtime.h>

#include "extent_cow.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

namespace {

constexpr std::size_t kPage = 4096U;

double elapsed_ms(const std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

char* make_region(const std::size_t bytes, const bool huge) {
  const std::size_t window = bytes + hostmm::kHugeBytes;
  void* const raw = ::mmap(nullptr, window, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) return nullptr;
  auto* const region = reinterpret_cast<char*>(
      (reinterpret_cast<std::uintptr_t>(raw) + hostmm::kHugeBytes - 1U) &
      ~(hostmm::kHugeBytes - 1U));
  if (::madvise(region, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) != 0) {
    return nullptr;
  }
  if (hostmm::populate_write(region, bytes, 12U) < 0.0) return nullptr;
  return region;
}

int fork_holder(const bool leaves_at_once) {
  int channel[2];
  if (::pipe(channel) != 0) return -1;
  const pid_t child = ::fork();
  if (child < 0) return -1;
  if (child == 0) {
    ::close(channel[1]);
    if (!leaves_at_once) {
      char go = 0;
      (void)!::read(channel[0], &go, 1);
    }
    ::_exit(0);
  }
  ::close(channel[0]);
  if (leaves_at_once) ::waitpid(child, nullptr, 0);
  return channel[1];
}

}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s {no_cuda|cuda} OPERATION MiB [THREADS]\n", argv[0]);
    return 2;
  }
  const std::string phase = argv[1];
  const std::string operation = argv[2];
  const std::size_t bytes =
      static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 10)) << 20U;
  const unsigned threads =
      argc > 4 ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 1U;
  if (bytes == 0U || bytes % hostmm::kHugeBytes != 0U || threads == 0U) return 2;

  const bool huge = operation == "thp_split" || operation == "fork_thp";
  char* const region = make_region(bytes, huge);
  if (region == nullptr) return 3;
  if (phase == "cuda") {
    if (cudaSetDevice(0) != cudaSuccess) return 3;

    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  } else if (phase != "no_cuda") {
    return 2;
  }

  const std::size_t pages = bytes / kPage;
  std::size_t events = 0U;
  const auto start = std::chrono::steady_clock::now();
  double ms = 0.0;
  if (operation == "cow_break") {

    const int holder = fork_holder(false);
    if (holder < 0) return 4;
    const auto break_start = std::chrono::steady_clock::now();
    if (hostmm::populate_write(region, bytes, threads) < 0.0) return 4;
    ms = elapsed_ms(break_start);
    ::close(holder);
    ::wait(nullptr);
    events = pages;
  } else if (operation == "reuse_fault") {

    if (fork_holder(true) < 0) return 4;
    const auto fault_start = std::chrono::steady_clock::now();
    if (hostmm::populate_write(region, bytes, threads) < 0.0) return 4;
    ms = elapsed_ms(fault_start);
    events = pages;
  } else if (operation == "fork" || operation == "fork_thp") {
    const pid_t child = ::fork();
    ms = elapsed_ms(start);
    if (child == 0) ::_exit(0);
    ::waitpid(child, nullptr, 0);
    events = 1U;
  } else if (operation == "mprotect_page") {

    for (std::size_t page = 0U; page < pages; ++page) {
      if (::mprotect(region + page * kPage, kPage, PROT_READ) != 0 ||
          ::mprotect(region + page * kPage, kPage, PROT_READ | PROT_WRITE) != 0) {
        return 4;
      }
    }
    ms = elapsed_ms(start);
    events = 2U * pages;
  } else if (operation == "mprotect_range") {
    if (::mprotect(region, bytes, PROT_READ) != 0 ||
        ::mprotect(region, bytes, PROT_READ | PROT_WRITE) != 0) {
      return 4;
    }
    ms = elapsed_ms(start);
    events = 2U;
  } else if (operation == "dontneed_page") {
    for (std::size_t page = 0U; page < pages; ++page) {
      if (::madvise(region + page * kPage, kPage, MADV_DONTNEED) != 0) return 4;
    }
    ms = elapsed_ms(start);
    events = pages;
  } else if (operation == "dontneed_range") {
    if (::madvise(region, bytes, MADV_DONTNEED) != 0) return 4;
    ms = elapsed_ms(start);
    events = 1U;
  } else if (operation == "munmap_range") {
    if (::munmap(region, bytes) != 0) return 4;
    ms = elapsed_ms(start);
    events = 1U;
  } else if (operation == "mremap_range") {
    void* const target = ::mmap(nullptr, bytes + hostmm::kHugeBytes, PROT_NONE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (target == MAP_FAILED) return 4;
    void* const aligned = reinterpret_cast<void*>(
        (reinterpret_cast<std::uintptr_t>(target) + hostmm::kHugeBytes - 1U) &
        ~(hostmm::kHugeBytes - 1U));
    const auto move_start = std::chrono::steady_clock::now();
    if (::mremap(region, bytes, bytes, MREMAP_MAYMOVE | MREMAP_FIXED, aligned) ==
        MAP_FAILED) {
      return 4;
    }
    ms = elapsed_ms(move_start);
    events = 1U;
  } else if (operation == "thp_split") {

    const std::size_t huge_pages = bytes / hostmm::kHugeBytes;
    for (std::size_t index = 0U; index < huge_pages; ++index) {
      if (::mprotect(region + index * hostmm::kHugeBytes, kPage, PROT_READ) != 0) {
        return 4;
      }
    }
    ms = elapsed_ms(start);
    events = huge_pages;
  } else if (operation == "thp_collapse") {

    if (::madvise(region, bytes, MADV_HUGEPAGE) != 0) return 4;
    const auto collapse_start = std::chrono::steady_clock::now();
    if (::madvise(region, bytes, MADV_COLLAPSE) != 0) return 5;
    ms = elapsed_ms(collapse_start);
    events = bytes / hostmm::kHugeBytes;
  } else {
    return 2;
  }
  std::printf("probe=mm_cost phase=%s operation=%s bytes=%zu threads=%u "
              "events=%zu ms=%.3f us_per_event=%.3f\n",
              phase.c_str(), operation.c_str(), bytes, threads, events, ms,
              1000.0 * ms / static_cast<double>(events));
  return 0;
}
