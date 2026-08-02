# 端侧性能分析

## 1. 如何采集

启动调试 daemon 后连续执行同一图片请求，以区分首次运行初始化和热运行：

```bash
./scripts/vlm_daemon.sh start

adb shell 'cd /data/local/tmp/mobile-nano-vlm && \
  ./minimind_cli client vision test_image.jpg "请描述这张图片"'
```

CLI 的 `Done` 行和 Android `MiniMindVLM` logcat 都会给出分阶段耗时。`total`
从服务端接受请求后开始计时，不包含 App UI 操作和模型加载；常驻模式下模型
加载只发生一次。

## 2. 当前实测结果

测试输入为 3000 × 4000 JPEG，设备为当前验证的 Snapdragon 8 Gen 3。连续
三次请求如下（单位 ms）：

| 次数 | total | image load | preprocess | Vision HTP | LLM | prefill | decode | token |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2366 | 790 | 15 | 5.8 | 1325 | 631 | 583 | 81+176 |
| 2 | 1901 | 840 | 15 | 4.8 | 811 | 49 | 642 | 81+196 |
| 3 | 1866 | 847 | 15 | 4.9 | 768 | 49 | 607 | 81+186 |

因此看到的约 2000 ms 不是 Vision 模型慢。Vision QNN/HTP forward 只有约
5 ms，主要耗时是两部分：

1. `stb_image` 在 CPU 上完整解码 3000 × 4000 JPEG，并复制为约 36 MB RGB
   缓冲，约 0.8 s。
2. LLM 逐 token 生成约 180～200 token，热运行 decode 约 0.6 s；首个请求
   还包含 QNN/MNN 延迟初始化，prefill 从热运行约 49 ms 增至约 631 ms。

`llm_ms` 大于 `prefill_ms + decode_ms` 是正常的：前者还覆盖采样、tokenizer
decode、流式输出及 MNN 调度等外围开销。各子项与 `total` 也不要求严格相加，
因为总耗时还覆盖临时缓冲释放和服务端协议处理。

## 3. 后续优化优先级

建议先建立固定图片、固定 prompt、固定最大输出长度的基准，再按顺序优化：

1. 图片路径：在不改变模型输入语义的前提下，避免完整 1200 万像素 JPEG
   解码后再缩到 256 × 256；评估相册缩略图、硬件解码或缩放解码。
2. LLM 输出：比较固定 token 数下的 tokens/s，减少不必要的生成长度，并分析
   decode 每 token 的 CPU/HTP 交互。
3. 首次请求：在服务 READY 后增加一次受控 warm-up，把 QNN 延迟初始化移出
   用户首个请求，并单独记录启动耗时。
4. 数据传输：用 QNN/MNN profiling 验证 CPU↔HTP copy，而不是只看端到端
   `total`。当前 Vision 仅约 5 ms，优先优化它的收益很小。

图片缩放优化必须做输出一致性回归。当前 native 预处理直接把原图双线性缩放
到 256 × 256；若 App 预先生成小 JPEG，会多一次压缩/解压并可能改变像素，不能
只凭耗时就替换生产路径。
