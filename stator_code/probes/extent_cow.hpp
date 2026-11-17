#pragma once

#include <cstddef>

namespace hostmm {

inline constexpr std::size_t kHugeBytes = std::size_t{2U} << 20U;

struct PrivatizeStats {
  double copy_ms = 0.0;
  double remap_ms = 0.0;
  std::size_t extents = 0U;
};

bool privatize_range(void* base, std::size_t bytes, std::size_t extent_bytes,
                     unsigned threads, bool huge_pages, PrivatizeStats* stats);

double populate_write(void* base, std::size_t bytes, unsigned threads);

}
