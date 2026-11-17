#include <llama.h>
#include "stator_timing.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <glob.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

std::string read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read " + path);
  }
  std::ostringstream text;
  text << input.rdbuf();
  return text.str();
}

std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& text, bool special) {
  const int needed = -llama_tokenize(vocab, text.data(),
                                     static_cast<int32_t>(text.size()),
                                     nullptr, 0, special, false);
  std::vector<llama_token> tokens(static_cast<size_t>(std::max(needed, 0)));
  const int written =
      llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()),
                     tokens.data(), static_cast<int32_t>(tokens.size()),
                     special, false);
  if (written != needed) {
    throw std::runtime_error("tokenization failed");
  }
  return tokens;
}

void decode(llama_context* context, std::vector<llama_token>& tokens,
            size_t first, size_t count, size_t batch) {
  for (size_t done = 0; done < count; done += batch) {
    const size_t piece = std::min(batch, count - done);
    if (llama_decode(context,
                     llama_batch_get_one(tokens.data() + first + done,
                                         static_cast<int32_t>(piece))) != 0) {
      throw std::runtime_error("decode failed");
    }
  }
  llama_synchronize(context);
}

std::string one_line(const std::string& text) {
  std::string line;
  for (const char c : text) {
    if (c == '\\') {
      line += "\\\\";
    } else if (c == '\n') {
      line += "\\n";
    } else if (c == '\r') {
      line += "\\r";
    } else {
      line += c;
    }
  }
  return line;
}

std::string task_of(int agent) {
  return "\n\nTask for agent " + std::to_string(agent) +
         ": summarize the text above in " + std::to_string(agent + 2) +
         " sentences.";
}

std::vector<std::string> child_environment(
    const std::vector<std::pair<std::string, std::string>>& changes) {
  std::vector<std::string> result;
  for (char** entry = environ; *entry != nullptr; ++entry) {
    const std::string line = *entry;
    const std::string name = line.substr(0, line.find('='));
    const bool replaced =
        std::any_of(changes.begin(), changes.end(),
                    [&name](const auto& change) { return change.first == name; });
    if (!replaced && name != "STATOR_KV_VMM_EXPORT") {
      result.push_back(line);
    }
  }
  for (const auto& [name, value] : changes) {
    result.push_back(name + "=" + value);
  }
  return result;
}

pid_t start_child(const std::string& binary,
                  const std::vector<std::string>& arguments,
                  const std::vector<std::string>& environment,
                  const std::string& output) {
  std::vector<char*> argv;
  for (const auto& argument : arguments) {
    argv.push_back(const_cast<char*>(argument.c_str()));
  }
  argv.push_back(nullptr);
  std::vector<char*> envp;
  for (const auto& entry : environment) {
    envp.push_back(const_cast<char*>(entry.c_str()));
  }
  envp.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, output.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
  const std::string errors = output + ".err";
  posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, errors.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC, 0644);
  pid_t pid = -1;
  const int status = posix_spawn(&pid, binary.c_str(), &actions, nullptr,
                                 argv.data(), envp.data());
  posix_spawn_file_actions_destroy(&actions);
  if (status != 0) {
    throw std::runtime_error("cannot start " + binary);
  }
  return pid;
}

int run(int argc, char** argv) {
  const auto process_start = Clock::now();
  stator_measure::Recorder trace;
  if (argc != 12) {
    std::fprintf(stderr,
                 "usage: fork_publisher MODEL CTX PREFIX_FILE STATE_BASE N_GEN FORKS "
                 "CHILDREN STEP OUT_DIR CHILD_BINARY GPU\n");
    return 2;
  }
  const std::string model_path = argv[1];
  const std::string context_text = argv[2];
  const uint32_t context_size =
      static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
  const std::string prefix_path = argv[3];
  const std::string state_base = argv[4];
  const std::string generate_text = argv[5];
  const int forks = std::atoi(argv[6]);
  const int children_per_fork = std::atoi(argv[7]);
  const int step = std::atoi(argv[8]);
  const std::string out_dir = argv[9];
  const std::string child_binary = argv[10];
  const std::string gpu = argv[11];
  const size_t batch = 2048;
  if (forks < 1 || children_per_fork < 1 || step < 1) {
    throw std::runtime_error("FORKS, CHILDREN and STEP must be positive");
  }

  trace.reserve(1024 + static_cast<size_t>(forks) * static_cast<size_t>(step + 64));
  llama_backend_init();
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 99;
  llama_model* model =
      llama_model_load_from_file(model_path.c_str(), model_params);
  if (model == nullptr) {
    throw std::runtime_error("cannot load " + model_path);
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int vocabulary = llama_vocab_n_tokens(vocab);
  llama_context_params context_params = llama_context_default_params();
  context_params.n_ctx = context_size;
  context_params.n_batch = static_cast<uint32_t>(batch);
  context_params.no_perf = true;
  llama_context* context = llama_init_from_model(model, context_params);
  if (context == nullptr) {
    throw std::runtime_error("cannot create the context");
  }

  std::vector<llama_token> tokens = tokenize(vocab, read_file(prefix_path), true);
  auto mark = Clock::now();
  trace.event("prefix_start");
  decode(context, tokens, 0, tokens.size(), batch);
  trace.event("prefix_end", static_cast<int64_t>(tokens.size()));
  const double prefix_ms = ms_since(mark);
  const std::vector<llama_token> task = tokenize(
      vocab, "\n\nTask for the parent: list the kernel mechanisms named above.",
      false);

  const char* host = std::getenv("STATOR_KV_HOST");
  const char* copy = std::getenv("STATOR_KV_PUBLISH_COPY");
  const bool copies = copy != nullptr && copy[0] == '1';
  const char* unlink_env = std::getenv("STATOR_REPEAT_UNLINK");
  const bool unlink_old = copies && host != nullptr && unlink_env != nullptr && unlink_env[0] == '1';
  std::vector<pid_t> children;
  std::string text;
  double publish_total_ms = 0.0;
  const auto first_publish = Clock::now();
  for (int fork = 1; fork <= forks; ++fork) {
    const std::string state_path = state_base + "." + std::to_string(fork);
    mark = Clock::now();
    trace.event("publish_start", fork);
    if (!llama_state_save_file(context, state_path.c_str(), tokens.data(),
                               tokens.size())) {
      throw std::runtime_error("cannot save the state " + state_path);
    }
    const double publish_ms = ms_since(mark);
    trace.event("publish_end", fork);
    publish_total_ms += publish_ms;
    std::printf("PUBLISHED fork=%d rows=%zu publish_ms=%.2f\n", fork,
                tokens.size(), publish_ms);
    std::fflush(stdout);

    std::vector<std::pair<std::string, std::string>> changes;
    if (host != nullptr) {
      changes.emplace_back("STATOR_KV_PREFIX", std::to_string(tokens.size()));
      if (copies && fork > 1) {
        changes.emplace_back("STATOR_KV_HOST",
                             std::string(host) + ".f" + std::to_string(fork));
      }
    }
    changes.emplace_back("CUDA_VISIBLE_DEVICES", gpu);
    for (int index = 0; index < children_per_fork; ++index) {
      const int agent = (fork - 1) * children_per_fork + index;
      children.push_back(start_child(
          child_binary,
          {child_binary, "child", model_path, context_text, state_path,
           task_of(agent), generate_text},
          child_environment(changes),
          out_dir + "/child." + std::to_string(fork) + "." + std::to_string(index)));
      trace.event("child_spawn", children.back(), agent);
    }

    if (unlink_old && fork > 1) {

      const std::string pattern = fork - 1 == 1 ? std::string(host) + ".[0-9]*"
                                                : std::string(host) + ".f" + std::to_string(fork - 1) + ".*";
      glob_t found{};
      if (glob(pattern.c_str(), 0, nullptr, &found) == 0) {
        for (size_t i = 0; i < found.gl_pathc; ++i) {
          unlink(found.gl_pathv[i]);
        }
      }
      globfree(&found);
      trace.event("unlink", fork - 1);
    }

    if (fork == 1) {
      tokens.insert(tokens.end(), task.begin(), task.end());
      decode(context, tokens, tokens.size() - task.size(), task.size(), batch);
    }
    for (int i = 0; i < step; ++i) {
      const float* logits = llama_get_logits_ith(context, -1);
      llama_token token = static_cast<llama_token>(
          std::max_element(logits, logits + vocabulary) - logits);
      char piece[256];
      const int length =
          llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
      if (length > 0) {
        text.append(piece, static_cast<size_t>(length));
      }
      tokens.push_back(token);
      if (llama_decode(context, llama_batch_get_one(&token, 1)) != 0) {
        throw std::runtime_error("decode failed");
      }
    }
    llama_synchronize(context);
  }
  const double parent_ms = ms_since(first_publish);

  for (size_t index = 0; index < children.size(); ++index) {
    int child_status = 0;
    waitpid(children[index], &child_status, 0);
    std::printf("CHILD index=%zu exit=%d\n", index,
                WIFEXITED(child_status) ? WEXITSTATUS(child_status) : 128);
  }
  std::printf(
      "RESULT role=parent forks=%d children=%zu prefix_ms=%.1f "
      "publish_total_ms=%.2f parent_ms=%.1f total_ms=%.1f rows=%zu\n",
      forks, children.size(), prefix_ms, publish_total_ms, parent_ms,
      ms_since(process_start), tokens.size());
  std::printf("TEXT %s\n", one_line(text).c_str());
  std::fflush(stdout);

  llama_free(context);
  llama_model_free(model);
  llama_backend_free();
  trace.finish();
  return 0;
}

}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "fork_publisher: %s\n", error.what());
    return 1;
  }
}
