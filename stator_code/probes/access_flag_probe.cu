#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MADV_COLD
#define MADV_COLD 20
#endif

namespace {

constexpr int kBlocks = 512;
constexpr int kThreads = 256;
constexpr float kOriginal = 3.0F;
constexpr std::size_t kPage = 4096U;
constexpr std::size_t kBlock = std::size_t{2U} << 20U;

__global__ void sum_kernel(const float* const data, const std::size_t count,
                           double* const partial) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  double total = 0.0;
  for (std::size_t index = start; index < count; index += stride) {
    total += static_cast<double>(data[index]);
  }
  partial[start] = total;
}

double anonymous_mib() {
  std::ifstream rollup("/proc/self/smaps_rollup");
  std::string key;
  while (rollup >> key) {
    if (key == "Anonymous:") {
      double kilobytes = 0.0;
      rollup >> kilobytes;
      return kilobytes / 1024.0;
    }
  }
  return -1.0;
}

}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr,
                 "usage: %s MiB {none|one_page|one_per_block|one_per_8_blocks|"
                 "all} {writable|readonly}\n",
                 argv[0]);
    return 2;
  }
  const std::size_t bytes =
      static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) << 20U;
  const std::string pattern = argv[2];
  const std::string protection = argv[3];
  const std::size_t count = bytes / sizeof(float);
  if (bytes == 0U || bytes % kBlock != 0U) return 2;

  const int fd = static_cast<int>(::syscall(SYS_memfd_create, "hostmm-af", 0U));
  if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(bytes)) != 0) return 2;
  auto* const staging = static_cast<float*>(
      ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  if (staging == MAP_FAILED) return 2;
  for (std::size_t index = 0U; index < count; ++index) staging[index] = kOriginal;
  ::munmap(staging, bytes);

  void* const raw = ::mmap(nullptr, bytes + kBlock, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (raw == MAP_FAILED) return 2;
  void* const base = reinterpret_cast<void*>(
      (reinterpret_cast<std::uintptr_t>(raw) + kBlock - 1U) & ~(kBlock - 1U));
  const int rights =
      protection == "readonly" ? PROT_READ : (PROT_READ | PROT_WRITE);
  auto* const model = static_cast<float*>(
      ::mmap(base, bytes, rights, MAP_PRIVATE | MAP_FIXED, fd, 0));
  if (model == MAP_FAILED) return 2;
  const auto* const view = reinterpret_cast<const volatile char*>(model);
  for (std::size_t offset = 0U; offset < bytes; offset += kPage) {
    (void)view[offset];
  }

  if (cudaSetDevice(0) != cudaSuccess) return 3;
  double* partial = nullptr;
  std::vector<double> host_partial(static_cast<std::size_t>(kBlocks) * kThreads);
  if (cudaMalloc(&partial, sizeof(double) * host_partial.size()) != cudaSuccess) {
    return 3;
  }
  const auto gpu_sum = [&](double* const ms) {
    const auto start = std::chrono::steady_clock::now();
    sum_kernel<<<kBlocks, kThreads>>>(model, count, partial);
    if (cudaDeviceSynchronize() != cudaSuccess) std::exit(4);
    *ms = std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start)
              .count();
    if (cudaMemcpy(host_partial.data(), partial,
                   host_partial.size() * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      std::exit(4);
    }
    double total = 0.0;
    for (const double value : host_partial) total += value;
    return total;
  };
  double warm_ms = 0.0, steady_ms = 0.0, after_ms = 0.0;
  gpu_sum(&warm_ms);
  const double before_sum = gpu_sum(&steady_ms);
  const double anonymous_before = anonymous_mib();

  std::size_t cleared = 0U;
  const std::size_t blocks = bytes / kBlock;
  const auto clear_page = [&](const std::size_t block) {
    char* const page = reinterpret_cast<char*>(model) + block * kBlock + 5U * kPage;
    if (::madvise(page, kPage, MADV_COLD) != 0) std::exit(5);
    ++cleared;
  };
  if (pattern == "one_page") {
    clear_page(blocks / 2U);
  } else if (pattern == "one_per_block") {
    for (std::size_t block = 0U; block < blocks; ++block) clear_page(block);
  } else if (pattern == "one_per_8_blocks") {
    for (std::size_t block = 0U; block < blocks; block += 8U) clear_page(block);
  } else if (pattern == "all") {
    if (::madvise(model, bytes, MADV_COLD) != 0) return 5;
    cleared = bytes / kPage;
  } else if (pattern != "none") {
    return 2;
  }

  const double after_sum = gpu_sum(&after_ms);
  const double growth = anonymous_mib() - anonymous_before;
  const double expected = static_cast<double>(count) * kOriginal;
  std::printf("probe=access_flag pattern=%s protection=%s bytes=%zu "
              "cleared_pages=%zu steady_read_ms=%.3f read_after_clear_ms=%.3f "
              "anonymous_growth_mib=%.3f copied_pages_per_cleared_page=%.1f "
              "sums_ok=%d\n",
              pattern.c_str(), protection.c_str(), bytes, cleared, steady_ms,
              after_ms, growth,
              cleared > 0U ? growth * 256.0 / static_cast<double>(cleared) : 0.0,
              before_sum == expected && after_sum == expected ? 1 : 0);
  return 0;
}
