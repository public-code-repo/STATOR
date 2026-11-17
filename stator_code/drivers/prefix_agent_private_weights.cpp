#include <llama.h>

namespace {

llama_model_params model_params_without_mmap() {
  llama_model_params params = llama_model_default_params();
  params.load_mode = LLAMA_LOAD_MODE_NONE;
  return params;
}

}

#define llama_model_default_params model_params_without_mmap
#include "prefix_agent.cpp"
