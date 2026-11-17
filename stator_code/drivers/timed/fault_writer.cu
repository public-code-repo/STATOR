#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <glob.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr std::uint64_t kBytes = 2ULL << 20U;

long long now_ns() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

__global__ void zero_words(std::uint32_t* data, std::uint64_t words) {
  const std::uint64_t thread =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread < words) data[thread] = 0U;
}

cudaError_t zero(void* address) {
  const std::uint64_t words = kBytes / sizeof(std::uint32_t);
  zero_words<<<static_cast<unsigned>((words + 255) / 256), 256>>>(
      static_cast<std::uint32_t*>(address), words);
  return cudaDeviceSynchronize();
}

bool ok(CUresult status) { return status == CUDA_SUCCESS; }

const char* name_of(CUresult status) {
  const char* text = nullptr;
  cuGetErrorName(status, &text);
  return text != nullptr ? text : "?";
}

CUresult set_access(CUdeviceptr address, size_t bytes, CUmemAccess_flags flags) {
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = 0;
  access.flags = flags;
  return cuMemSetAccess(address, bytes, &access, 1U);
}

void report(const char* route, long long start, size_t targets, size_t written,
            const char* result) {
  std::printf("WRITE route=%s start_ns=%lld end_ns=%lld targets=%zu written=%zu "
              "result=%s\n",
              route, start, now_ns(), targets, written, result);
  std::fflush(stdout);
}

int vmm(const char* import) {
  std::vector<int> fds;
  char* end = nullptr;
  std::strtol(import, &end, 10);
  for (const char* at = end; *at != '\0'; at = end) {
    const long value = std::strtol(at, &end, 10);
    if (end == at) break;
    fds.push_back(static_cast<int>(value));
  }

  std::vector<std::pair<CUdeviceptr, size_t>> ranges;
  const char* failure = "CUDA_SUCCESS";
  for (const int fd : fds) {
    CUmemGenericAllocationHandle handle = 0;
    CUresult status = cuMemImportFromShareableHandle(
        &handle, reinterpret_cast<void*>(static_cast<uintptr_t>(fd)),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    CUdeviceptr range = 0;
    size_t size = 0;
    for (size_t step = 1; ok(status) && step <= 1024 && size == 0; ++step) {
      CUdeviceptr tried = 0;
      status = cuMemAddressReserve(&tried, step * kBytes, 0U, 0U, 0ULL);
      if (!ok(status)) break;
      if (ok(cuMemMap(tried, step * kBytes, 0U, handle, 0ULL))) {
        range = tried;
        size = step * kBytes;
      } else {
        cuMemAddressFree(tried, step * kBytes);
      }
    }
    if (ok(status) && size == 0) status = CUDA_ERROR_INVALID_VALUE;
    if (ok(status)) status = set_access(range, size, CU_MEM_ACCESS_FLAGS_PROT_READ);
    if (!ok(status)) {
      failure = name_of(status);
      break;
    }
    ranges.emplace_back(range, size);
  }
  const char* delay = std::getenv("STATOR_FAULT_DELAY_MS");
  usleep(static_cast<useconds_t>(delay != nullptr ? std::atol(delay) : 0) * 1000U);
  const long long start = now_ns();
  size_t written = 0;
  for (const auto& [range, size] : ranges) {
    const CUresult raise = set_access(range, size, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    if (!ok(raise)) {
      failure = name_of(raise);
      break;
    }
    const cudaError_t status = zero(reinterpret_cast<void*>(range));
    if (status != cudaSuccess) {
      failure = cudaGetErrorName(status);
      break;
    }
    ++written;
  }
  report("vmm", start, fds.size(), written, failure);
  _exit(0);
}

int host(const char* base) {
  const std::string pattern = std::string(base) + ".*";
  glob_t found{};
  if (glob(pattern.c_str(), 0, nullptr, &found) != 0) {
    report("host", now_ns(), 0, 0, "no_file");
    return 3;
  }
  std::vector<void*> views;
  for (size_t i = 0; i < found.gl_pathc; ++i) {
    const int fd = open(found.gl_pathv[i], O_RDONLY);
    struct stat status {};
    if (fd < 0 || fstat(fd, &status) != 0 ||
        static_cast<std::uint64_t>(status.st_size) < kBytes) {
      continue;
    }
    void* mapped = mmap(nullptr, kBytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) continue;

    const volatile char* view = static_cast<const volatile char*>(mapped);
    for (std::uint64_t offset = 0; offset < kBytes; offset += 4096) (void)view[offset];
    views.push_back(mapped);
  }
  const char* delay = std::getenv("STATOR_FAULT_DELAY_MS");
  usleep(static_cast<useconds_t>(delay != nullptr ? std::atol(delay) : 0) * 1000U);
  const long long start = now_ns();
  size_t written = 0;
  const char* failure = "cudaSuccess";
  for (void* view : views) {
    const cudaError_t status = zero(view);
    if (status != cudaSuccess) {
      failure = cudaGetErrorName(status);
      break;
    }
    ++written;
  }
  report("host", start, views.size(), written, failure);

  _exit(0);
}

}

int main() {
  CUdevice device = 0;
  CUcontext context = nullptr;
  if (!ok(cuInit(0U)) || !ok(cuDeviceGet(&device, 0)) ||
      !ok(cuDevicePrimaryCtxRetain(&context, device)) ||
      !ok(cuCtxSetCurrent(context)) || cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "fault_writer: no CUDA context\n");
    return 3;
  }
  const char* import = std::getenv("STATOR_KV_VMM_IMPORT");
  const char* base = std::getenv("STATOR_KV_HOST");
  if (import != nullptr && import[0] != '\0') return vmm(import);
  if (base != nullptr && base[0] != '\0') return host(base);
  std::fprintf(stderr, "fault_writer: neither STATOR_KV_VMM_IMPORT nor STATOR_KV_HOST\n");
  return 2;
}
