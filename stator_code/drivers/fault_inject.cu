#include <cuda_runtime.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

namespace {

__global__ void fault_write(unsigned* page) { page[0] = 0x5eedf00dU; }

double now_s() {
  timespec ts{};
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

void* injector(void* argument) {
  const char* trigger = static_cast<const char*>(argument);
  while (access(trigger, F_OK) != 0) usleep(10000);
  void* page = mmap(nullptr, 4096, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (page == MAP_FAILED) {
    std::fprintf(stderr, "FAULT_INJECTED status=mmap_failed\n");
    return nullptr;
  }
  const double start = now_s();
  fault_write<<<1, 1>>>(static_cast<unsigned*>(page));
  const cudaError_t launch = cudaGetLastError();
  const cudaError_t status = cudaDeviceSynchronize();
  std::fprintf(stderr, "FAULT_INJECTED at_s=%.3f launch=%s status=%s\n", start,
               cudaGetErrorName(launch), cudaGetErrorName(status));
  std::fflush(stderr);
  return nullptr;
}

__attribute__((constructor)) void start_injector() {
  const char* value = std::getenv("FAULT_TRIGGER");
  if (value == nullptr || value[0] == '\0') return;

  static char trigger[4096];
  std::snprintf(trigger, sizeof(trigger), "%s", value);
  unsetenv("FAULT_TRIGGER");
  unsetenv("LD_PRELOAD");
  pthread_t thread;
  if (pthread_create(&thread, nullptr, injector, trigger) == 0) {
    pthread_detach(thread);
  }
}

}
