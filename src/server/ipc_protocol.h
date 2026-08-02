#pragma once

#include <cstddef>
#include <string>

namespace minimind::ipc {

constexpr std::size_t kMaxFrameSize = 1024 * 1024;

bool read_frame(int fd, std::string* payload, std::string* error);
bool write_frame(int fd, const std::string& payload, std::string* error);

int create_unix_server(const std::string& socket_path, std::string* error);
int connect_unix_socket(const std::string& socket_path, std::string* error);

}  // namespace minimind::ipc
