#include <cuda_runtime.h>

#include "extent_cow.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {

__global__ void fill_kernel(float* const data, const std::size_t count,
                            const float value) {
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index = start; index < count; index += stride) {
    data[index] = value;
  }
}

double elapsed_ms(const std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

double gpu_write_ms(float* const data, const std::size_t count,
                    const float value) {
  const auto start = std::chrono::steady_clock::now();
  fill_kernel<<<512, 256>>>(data, count, value);
  if (cudaDeviceSynchronize() != cudaSuccess) std::exit(4);
  return elapsed_ms(start);
}

bool fork_exec_helper() {
  const pid_t child = ::fork();
  if (child < 0) return false;
  if (child == 0) {
    ::execl("/bin/true", "true", static_cast<char*>(nullptr));
    ::_exit(127);
  }
  int status = 0;
  ::waitpid(child, &status, 0);
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s {fork|dontfork|posix_spawn|repair} MiB\n",
                 argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  const std::size_t bytes =
      static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)) << 20U;
  const std::size_t count = bytes / sizeof(float);
  const std::size_t window = bytes + hostmm::kHugeBytes;
  void* const raw = ::mmap(nullptr, window, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) return 2;
  auto* const data = reinterpret_cast<float*>(
      (reinterpret_cast<std::uintptr_t>(raw) + hostmm::kHugeBytes - 1U) &
      ~(hostmm::kHugeBytes - 1U));
  (void)::madvise(data, bytes, MADV_HUGEPAGE);
  if (hostmm::populate_write(data, bytes, 12U) < 0.0) return 2;
  if (cudaSetDevice(0) != cudaSuccess) return 3;
  gpu_write_ms(data, count, 1.0F);
  const double steady = gpu_write_ms(data, count, 1.0F);

  const auto start = std::chrono::steady_clock::now();
  bool started = false;
  if (mode == "fork" || mode == "repair") {
    started = fork_exec_helper();
    if (started && mode == "repair") {
      started = hostmm::privatize_range(data, bytes, bytes, 12U, true, nullptr);
    }
  } else if (mode == "dontfork") {
    if (::madvise(data, bytes, MADV_DONTFORK) != 0) return 5;
    started = fork_exec_helper();
    if (::madvise(data, bytes, MADV_DOFORK) != 0) return 5;
  } else if (mode == "posix_spawn") {
    pid_t child = -1;
    char name[] = "true";
    char* const arguments[] = {name, nullptr};
    if (::posix_spawn(&child, "/bin/true", nullptr, nullptr, arguments,
                      environ) == 0) {
      int status = 0;
      ::waitpid(child, &status, 0);
      started = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
  } else {
    return 2;
  }
  const double helper_ms = elapsed_ms(start);
  if (!started) return 5;

  const double first = gpu_write_ms(data, count, 2.0F);
  const double next = gpu_write_ms(data, count, 3.0F);
  std::size_t wrong = 0U;
  for (std::size_t index = 0U; index < count; index += 257U) {
    wrong += data[index] != 3.0F ? 1U : 0U;
  }
  std::printf("probe=fork_probe mode=%s bytes=%zu steady_gpu_write_ms=%.3f "
              "helper_ms=%.3f first_gpu_write_after_helper_ms=%.3f "
              "next_gpu_write_ms=%.3f ok=%d\n",
              mode.c_str(), bytes, steady, helper_ms, first, next,
              wrong == 0U ? 1 : 0);
  return wrong == 0U ? 0 : 1;
}
