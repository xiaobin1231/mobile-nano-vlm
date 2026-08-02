#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "server/vlm_service.h"

#include "common/stb_image.h"
#include "pipeline/pipeline.h"
#include "server/ipc_protocol.h"
#include "ujson.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <ostream>
#include <streambuf>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace minimind {
namespace {

volatile std::sig_atomic_t g_stop_requested = 0;
volatile std::sig_atomic_t g_listen_fd = -1;
constexpr char kVlmSocketName[] = "@minimind_vlm";
constexpr int kDefaultMaxTokens = 512;

void handle_signal(int) {
    g_stop_requested = 1;
    const int fd = g_listen_fd;
    if (fd >= 0) close(fd);
}

bool send_json(int fd, const ujson::json& message) {
    std::string error;
    if (!ipc::write_frame(fd, message.dump(), &error)) {
        std::cerr << "Socket write failed: " << error << std::endl;
        return false;
    }
    return true;
}

ujson::json make_message(const std::string& type,
                         const std::string& request_id) {
    ujson::json message;
    message["version"] = 1;
    message["type"] = type;
    message["request_id"] = request_id;
    return message;
}

bool send_error(int fd, const std::string& request_id,
                const std::string& detail) {
    auto message = make_message("error", request_id);
    message["message"] = detail;
    return send_json(fd, message);
}

// MNN may flush tokenizer byte pieces before a complete UTF-8 code point has
// been produced. Keep an incomplete suffix for the next flush so every JSON
// token frame contains valid UTF-8. Terminal clients happened to concatenate
// the raw frames before rendering, while Android decoded each frame separately
// and displayed U+FFFD for a split Chinese character.
std::string take_complete_utf8(std::string* buffer) {
    std::string output;
    output.reserve(buffer->size());
    std::size_t index = 0;
    while (index < buffer->size()) {
        const auto first = static_cast<unsigned char>((*buffer)[index]);
        if (first <= 0x7f) {
            output.push_back((*buffer)[index++]);
            continue;
        }

        std::size_t length = 0;
        if (first >= 0xc2 && first <= 0xdf) {
            length = 2;
        } else if (first >= 0xe0 && first <= 0xef) {
            length = 3;
        } else if (first >= 0xf0 && first <= 0xf4) {
            length = 4;
        } else {
            output.append("\xef\xbf\xbd");
            ++index;
            continue;
        }

        if (buffer->size() - index < length) break;
        bool valid = true;
        for (std::size_t offset = 1; offset < length; ++offset) {
            const auto continuation =
                static_cast<unsigned char>((*buffer)[index + offset]);
            if ((continuation & 0xc0) != 0x80) {
                valid = false;
                break;
            }
        }
        const auto second = static_cast<unsigned char>((*buffer)[index + 1]);
        if ((first == 0xe0 && second < 0xa0)
            || (first == 0xed && second >= 0xa0)
            || (first == 0xf0 && second < 0x90)
            || (first == 0xf4 && second >= 0x90)) {
            valid = false;
        }
        if (!valid) {
            output.append("\xef\xbf\xbd");
            ++index;
            continue;
        }

        output.append(*buffer, index, length);
        index += length;
    }
    buffer->erase(0, index);
    return output;
}

std::vector<uint8_t> load_image(const std::string& path, int* width,
                                int* height, std::string* error) {
    int channels = 0;
    uint8_t* data = stbi_load(path.c_str(), width, height, &channels, 3);
    if (!data) {
        if (error) {
            *error = "failed to load image: " + path + " ("
                + (stbi_failure_reason() ? stbi_failure_reason() : "unknown") + ")";
        }
        return {};
    }
    std::vector<uint8_t> result(data, data + (*width) * (*height) * 3);
    stbi_image_free(data);
    return result;
}

class TokenStreamBuffer final : public std::streambuf {
public:
    TokenStreamBuffer(int fd, std::string request_id)
        : fd_(fd), request_id_(std::move(request_id)) {}

    bool healthy() const { return healthy_; }

    bool finish() {
        if (sync() != 0) return false;
        if (!pending_.empty()) {
            // A well-formed model response should never end mid-code-point.
            // Emit one replacement character instead of dropping bytes or
            // placing invalid UTF-8 inside the final JSON frame.
            pending_.clear();
            return send_text("\xef\xbf\xbd");
        }
        return healthy_;
    }

protected:
    std::streamsize xsputn(const char* data, std::streamsize size) override {
        if (size > 0) pending_.append(data, static_cast<std::size_t>(size));
        return size;
    }

    int overflow(int character) override {
        if (character != traits_type::eof()) {
            pending_.push_back(static_cast<char>(character));
        }
        return traits_type::not_eof(character);
    }

    int sync() override {
        if (pending_.empty()) return healthy_ ? 0 : -1;
        const std::string complete = take_complete_utf8(&pending_);
        if (complete.empty()) return healthy_ ? 0 : -1;
        return send_text(complete) ? 0 : -1;
    }

private:
    bool send_text(const std::string& text) {
        auto message = make_message("token", request_id_);
        message["text"] = text;
        healthy_ = send_json(fd_, message);
        return healthy_;
    }

    int fd_;
    std::string request_id_;
    std::string pending_;
    bool healthy_ = true;
};

bool peer_is_allowed(int fd) {
#ifdef SO_PEERCRED
    struct ucred peer {};
    socklen_t length = sizeof(peer);
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &peer, &length) != 0) {
        return false;
    }
    return peer.uid == getuid();
#else
    (void)fd;
    return true;
#endif
}

class VlmServer {
public:
    VlmServer(std::string config_dir, std::string vision_model,
              std::string socket_path, int num_threads)
        : config_dir_(std::move(config_dir)),
          vision_model_(std::move(vision_model)),
          socket_path_(std::move(socket_path)),
          num_threads_(num_threads) {}

    ~VlmServer() {
        if (listen_fd_ >= 0) close(listen_fd_);
        if (!socket_path_.empty() && socket_path_.front() != '@') {
            unlink(socket_path_.c_str());
        }
    }

    int run() {
        std::cout << "Loading resident VLM pipeline..." << std::endl;
        const auto load_start = std::chrono::steady_clock::now();
        if (!pipeline_.load(vision_model_, config_dir_, num_threads_)) {
            std::cerr << "Pipeline load failed" << std::endl;
            return 1;
        }
        const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - load_start).count();

        std::string error;
        listen_fd_ = ipc::create_unix_server(socket_path_, &error);
        if (listen_fd_ < 0) {
            std::cerr << "Server socket failed: " << error << std::endl;
            return 1;
        }
        g_listen_fd = listen_fd_;
        std::cout << "READY load_ms=" << load_ms << std::endl;

        while (!g_stop_requested && !shutdown_requested_) {
            const int client_fd = accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
            if (client_fd < 0) {
                if (errno == EINTR) continue;
                if (g_stop_requested) break;
                std::cerr << "accept failed: " << std::strerror(errno) << std::endl;
                continue;
            }
            if (!peer_is_allowed(client_fd)) {
                send_error(client_fd, "", "client UID is not allowed");
                close(client_fd);
                continue;
            }
            handle_client(client_fd);
            close(client_fd);
        }
        std::cout << "VLM server stopped" << std::endl;
        return 0;
    }

private:
    void handle_client(int fd) {
        while (!g_stop_requested && !shutdown_requested_) {
            std::string payload;
            std::string error;
            if (!ipc::read_frame(fd, &payload, &error)) return;

            try {
                const auto request = ujson::json::parse(payload);
                const std::string type = request.value("type", "");
                const std::string request_id = request.value("request_id", "");

                if (type == "ping") {
                    auto response = make_message("ready", request_id);
                    response["model_loaded"] = pipeline_.ready();
                    if (!send_json(fd, response)) return;
                    continue;
                }
                if (type == "shutdown") {
                    auto response = make_message("stopping", request_id);
                    send_json(fd, response);
                    shutdown_requested_ = true;
                    return;
                }
                if (type != "generate") {
                    if (!send_error(fd, request_id, "unsupported request type")) return;
                    continue;
                }

                const std::string prompt = request.value("prompt", "");
                const std::string image_path = request.value("image_path", "");
                if (prompt.empty()) {
                    if (!send_error(fd, request_id, "prompt is empty")) return;
                    continue;
                }

                auto accepted = make_message("accepted", request_id);
                accepted["has_image"] = !image_path.empty();
                if (!send_json(fd, accepted)) return;

                const auto start = std::chrono::steady_clock::now();
                TokenStreamBuffer stream_buffer(fd, request_id);
                std::ostream token_stream(&stream_buffer);
                bool ok = false;
                double image_load_ms = 0.0;
                if (image_path.empty()) {
                    ok = pipeline_.run_text(
                        prompt, token_stream, kDefaultMaxTokens);
                } else {
                    int width = 0;
                    int height = 0;
                    const auto image_load_start = std::chrono::steady_clock::now();
                    auto image = load_image(image_path, &width, &height, &error);
                    image_load_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - image_load_start).count();
                    if (image.empty()) {
                        send_error(fd, request_id, error);
                        continue;
                    }
                    ok = pipeline_.run_vision(image.data(), width, height,
                                              prompt, token_stream,
                                              kDefaultMaxTokens);
                }
                stream_buffer.finish();
                if (!ok || !stream_buffer.healthy()) {
                    if (stream_buffer.healthy()) {
                        send_error(fd, request_id, "inference failed");
                    }
                    continue;
                }

                const auto elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count();
                auto done = make_message("done", request_id);
                done["elapsed_ms"] = static_cast<int64_t>(elapsed_ms);
                const auto& profile = pipeline_.last_profile();
                done["image_load_ms"] = image_load_ms;
                done["preprocess_ms"] = profile.preprocess_ms;
                done["vision_ms"] = profile.vision_ms;
                done["tokenize_ms"] = profile.tokenize_ms;
                done["embedding_ms"] = profile.embedding_ms;
                done["mix_ms"] = profile.mix_ms;
                done["llm_ms"] = profile.llm_ms;
                done["prefill_ms"] = profile.prefill_ms;
                done["decode_ms"] = profile.decode_ms;
                done["prompt_tokens"] = profile.prompt_tokens;
                done["generated_tokens"] = profile.generated_tokens;
                if (!send_json(fd, done)) return;
            } catch (const std::exception& exception) {
                if (!send_error(fd, "", std::string("invalid request: ")
                                + exception.what())) {
                    return;
                }
            }
        }
    }

    std::string config_dir_;
    std::string vision_model_;
    std::string socket_path_;
    int num_threads_ = 4;
    int listen_fd_ = -1;
    bool shutdown_requested_ = false;
    Pipeline pipeline_;
};

std::string make_request_id() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::to_string(getpid()) + "-" + std::to_string(now);
}

}  // namespace

int run_vlm_server(const std::string& config_dir,
                   const std::string& vision_model,
                   int num_threads) {
    g_stop_requested = 0;
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    VlmServer server(config_dir, vision_model, kVlmSocketName,
                     std::max(1, num_threads));
    return server.run();
}

int run_vlm_client(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " client ping\n"
                  << "  " << argv[0] << " client shutdown\n"
                  << "  " << argv[0] << " client text <prompt>\n"
                  << "  " << argv[0] << " client vision <image.jpg> <prompt>\n";
        return 1;
    }

    const std::string action = argv[2];
    const std::string request_id = make_request_id();
    ujson::json request = make_message("", request_id);

    if (action == "ping" || action == "shutdown") {
        request["type"] = action;
    } else if (action == "text") {
        if (argc < 4) {
            std::cerr << "text requires a prompt" << std::endl;
            return 1;
        }
        request["type"] = "generate";
        request["prompt"] = argv[3];
        request["image_path"] = "";
    } else if (action == "vision") {
        if (argc < 5) {
            std::cerr << "vision requires an image and prompt" << std::endl;
            return 1;
        }
        request["type"] = "generate";
        request["image_path"] = argv[3];
        request["prompt"] = argv[4];
    } else {
        std::cerr << "unsupported client action: " << action << std::endl;
        return 1;
    }

    std::string error;
    const int fd = ipc::connect_unix_socket(kVlmSocketName, &error);
    if (fd < 0) {
        std::cerr << "Connect failed: " << error << std::endl;
        return 1;
    }
    if (!ipc::write_frame(fd, request.dump(), &error)) {
        std::cerr << "Request failed: " << error << std::endl;
        close(fd);
        return 1;
    }

    int result = 1;
    while (true) {
        std::string payload;
        if (!ipc::read_frame(fd, &payload, &error)) {
            std::cerr << "Response failed: " << error << std::endl;
            break;
        }
        try {
            const auto response = ujson::json::parse(payload);
            const std::string type = response.value("type", "");
            if (type == "token") {
                std::cout << response.value("text", "") << std::flush;
            } else if (type == "done") {
                std::cout << std::endl;
                std::cerr
                    << "Done: total=" << response.value("elapsed_ms", 0) << " ms"
                    << ", image_load=" << response.value("image_load_ms", 0.0) << " ms"
                    << ", preprocess=" << response.value("preprocess_ms", 0.0) << " ms"
                    << ", vision=" << response.value("vision_ms", 0.0) << " ms"
                    << ", tokenize=" << response.value("tokenize_ms", 0.0) << " ms"
                    << ", embedding=" << response.value("embedding_ms", 0.0) << " ms"
                    << ", mix=" << response.value("mix_ms", 0.0) << " ms"
                    << ", llm=" << response.value("llm_ms", 0.0) << " ms"
                    << " (prefill=" << response.value("prefill_ms", 0.0) << " ms"
                    << ", decode=" << response.value("decode_ms", 0.0) << " ms)"
                    << ", tokens=" << response.value("prompt_tokens", 0)
                    << "+" << response.value("generated_tokens", 0)
                    << std::endl;
                result = 0;
                break;
            } else if (type == "ready") {
                std::cout << "READY" << std::endl;
                result = 0;
                break;
            } else if (type == "stopping") {
                std::cout << "STOPPING" << std::endl;
                result = 0;
                break;
            } else if (type == "error") {
                std::cerr << "Server error: "
                          << response.value("message", "unknown") << std::endl;
                break;
            }
        } catch (const std::exception& exception) {
            std::cerr << "Invalid response: " << exception.what() << std::endl;
            break;
        }
    }
    close(fd);
    return result;
}

}  // namespace minimind
