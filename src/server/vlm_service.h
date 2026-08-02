#pragma once

#include <string>

namespace minimind {

int run_vlm_server(const std::string& config_dir,
                   const std::string& vision_model,
                   int num_threads);

int run_vlm_client(int argc, char* argv[]);

}  // namespace minimind
