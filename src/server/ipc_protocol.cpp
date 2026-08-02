#include "server/ipc_protocol.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace minimind::ipc {
namespace {

bool read_all(int fd, void* data, std::size_t size, std::string* error) {
    auto* cursor = static_cast<unsigned char*>(data);
    while (size > 0) {
        const ssize_t count = recv(fd, cursor, size, 0);
        if (count == 0) {
            if (error) *error = "peer closed connection";
            return false;
        }
        if (count < 0) {
            if (errno == EINTR) continue;
            if (error) *error = std::strerror(errno);
            return false;
        }
        cursor += count;
        size -= static_cast<std::size_t>(count);
    }
    return true;
}

bool write_all(int fd, const void* data, std::size_t size, std::string* error) {
    const auto* cursor = static_cast<const unsigned char*>(data);
    while (size > 0) {
        const ssize_t count = send(fd, cursor, size, MSG_NOSIGNAL);
        if (count < 0) {
            if (errno == EINTR) continue;
            if (error) *error = std::strerror(errno);
            return false;
        }
        cursor += count;
        size -= static_cast<std::size_t>(count);
    }
    return true;
}

bool fill_address(const std::string& path, sockaddr_un* address,
                  socklen_t* address_size, std::string* error) {
    if (path.empty()) {
        if (error) *error = "socket path is empty";
        return false;
    }
    const bool abstract = path.front() == '@';
    const std::size_t name_size = path.size() - (abstract ? 1 : 0);
    if (name_size + (abstract ? 1 : 0) >= sizeof(address->sun_path)) {
        if (error) *error = "socket path is too long";
        return false;
    }
    std::memset(address, 0, sizeof(*address));
    address->sun_family = AF_UNIX;
    if (abstract) {
        address->sun_path[0] = '\0';
        std::memcpy(address->sun_path + 1, path.data() + 1, name_size);
        *address_size = static_cast<socklen_t>(
            offsetof(sockaddr_un, sun_path) + 1 + name_size);
    } else {
        std::memcpy(address->sun_path, path.c_str(), path.size() + 1);
        *address_size = static_cast<socklen_t>(
            offsetof(sockaddr_un, sun_path) + path.size() + 1);
    }
    return true;
}

}  // namespace

bool read_frame(int fd, std::string* payload, std::string* error) {
    if (!payload) return false;
    uint32_t network_size = 0;
    if (!read_all(fd, &network_size, sizeof(network_size), error)) return false;
    const std::size_t size = ntohl(network_size);
    if (size == 0 || size > kMaxFrameSize) {
        if (error) *error = "invalid frame size: " + std::to_string(size);
        return false;
    }
    payload->resize(size);
    return read_all(fd, payload->data(), size, error);
}

bool write_frame(int fd, const std::string& payload, std::string* error) {
    if (payload.empty() || payload.size() > kMaxFrameSize) {
        if (error) *error = "invalid frame size: " + std::to_string(payload.size());
        return false;
    }
    const uint32_t network_size = htonl(static_cast<uint32_t>(payload.size()));
    return write_all(fd, &network_size, sizeof(network_size), error)
        && write_all(fd, payload.data(), payload.size(), error);
}

int create_unix_server(const std::string& socket_path, std::string* error) {
    sockaddr_un address{};
    socklen_t address_size = 0;
    if (!fill_address(socket_path, &address, &address_size, error)) return -1;

    const bool abstract = socket_path.front() == '@';
    if (!abstract) {
        struct stat info {};
        if (lstat(socket_path.c_str(), &info) == 0) {
            if (!S_ISSOCK(info.st_mode)) {
                if (error) *error = "refusing to replace non-socket path";
                return -1;
            }
            if (unlink(socket_path.c_str()) != 0) {
                if (error) *error = std::strerror(errno);
                return -1;
            }
        } else if (errno != ENOENT) {
            if (error) *error = std::strerror(errno);
            return -1;
        }
    }

    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        if (error) *error = std::strerror(errno);
        return -1;
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&address), address_size) != 0
        || (!abstract && chmod(socket_path.c_str(), 0600) != 0)
        || listen(fd, 4) != 0) {
        if (error) *error = std::strerror(errno);
        close(fd);
        if (!abstract) unlink(socket_path.c_str());
        return -1;
    }
    return fd;
}

int connect_unix_socket(const std::string& socket_path, std::string* error) {
    sockaddr_un address{};
    socklen_t address_size = 0;
    if (!fill_address(socket_path, &address, &address_size, error)) return -1;
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        if (error) *error = std::strerror(errno);
        return -1;
    }
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), address_size) != 0) {
        if (error) *error = std::strerror(errno);
        close(fd);
        return -1;
    }
    return fd;
}

}  // namespace minimind::ipc
