# 常驻 VLM 推理服务

`minimind_cli server` 在启动时加载一次 Vision、LLM、MNN Wrapper 和 QNN HTP
Context，之后通过 Unix Domain Socket 串行处理纯文本或图文请求。协议版本 1
中的每个请求相互独立，不复用上一轮的 KV Cache。

## 1. 启动与管理

完成 Android 编译和部署后，在主机执行：

```bash
./scripts/vlm_daemon.sh start
./scripts/vlm_daemon.sh status
./scripts/vlm_daemon.sh logs
./scripts/vlm_daemon.sh stop
```

默认参数：

```text
DEVICE_ROOT=/data/local/tmp/mobile-nano-vlm
VLM_THREADS=4
VLM_LOG_FILE=vlm_server.log
```

可在启动前通过同名环境变量覆盖。服务内部使用 Android 抽象命名空间 Unix
Domain Socket，地址属于内部实现，不需要用户配置或传入。

## 2. adb Client 验证

```bash
adb shell
cd /data/local/tmp/mobile-nano-vlm
```

健康检查：

```sh
./minimind_cli client ping
```

纯文本：

```sh
./minimind_cli client \
  text '请简单介绍你自己'
```

图文请求：

```sh
./minimind_cli client \
  vision test_image.jpg '请描述这张图片'
```

服务端固定最多生成 512 个新 token，用户无需设置该参数。模型生成结束 token
时仍会提前停止。

## 3. 协议

每条消息由“4 字节大端 JSON 长度 + UTF-8 JSON”组成，最大 Frame 为 1 MiB。
生成请求示例：

```json
{
  "version": 1,
  "type": "generate",
  "request_id": "client-42",
  "prompt": "请描述这张图片",
  "image_path": "/data/local/tmp/mobile-nano-vlm/test_image.jpg"
}
```

服务端按顺序返回：

```json
{"version":1,"type":"accepted","request_id":"client-42"}
{"version":1,"type":"token","request_id":"client-42","text":"这"}
{"version":1,"type":"token","request_id":"client-42","text":"张图片"}
{"version":1,"type":"done","request_id":"client-42","elapsed_ms":1901,
 "image_load_ms":839.6,"preprocess_ms":15.0,"vision_ms":4.8,
 "llm_ms":811.0,"prefill_ms":49.5,"decode_ms":642.4,
 "prompt_tokens":81,"generated_tokens":196}
```

`done` 中的性能字段用于区分图片解码、Vision、LLM prefill/decode 等阶段。
详细口径和当前实测数据见[端侧性能分析](performance.md)。

控制消息的 `type` 为 `ping` 或 `shutdown`。服务端使用 `SO_PEERCRED`，只接受
与自身 UID 相同的客户端。

## 4. 当前并发和生命周期

- Pipeline 在 server 启动时加载一次，请求期间不会重复加载模型。
- 请求串行执行，避免多个线程同时访问同一 MNN Session、KV Cache 和 QNN
  HTP Context。
- Client 断开不会卸载模型；重新连接后可以继续请求。
- `SIGINT`、`SIGTERM` 或 `shutdown` 请求会让 server 正常退出。
- `scripts/vlm_daemon.sh` 用于 adb 开发调试。Android App 已由 Foreground
  Service 以 App UID 启动 server，并用 Android `LocalSocket` 连接抽象
  Socket，详见[Android App 常驻服务集成](android-app-integration.md)。

## 5. 流式输出实现

工程没有修改 `third_party/MNN`。`LlmRuntime` 调用 MNN 已有的
`Llm::response(..., std::ostream*)` 接口，自定义 `std::streambuf` 在 MNN 每次
flush 时发送一条 `token` Frame。每个请求开始前显式执行 `Llm::reset()`，防止
上一轮 history、输出 token 或 KV 状态泄漏到下一轮。
