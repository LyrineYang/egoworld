# 有预算的共享采样帧与 crop 缓存

这是 `design.md` 第五部分 B 的首版实现。缓存服务于**同一 run 内、同一准备片段的输入复用**，
由 HaMeR、FoundationPose 和 Fast3R 共用。它不保存模型权重、跟踪器状态或阶段输出；
阶段 Parquet/提交标记仍承担恢复职责。产品入口的跨 run 阶段结果复用是另一层机制。

## 显式启用

默认关闭。用户可以在自己的配置中选择：

```json
{
  "extra": {
    "input_cache": {
      "enabled": true,
      "max_disk_bytes": 536870912,
      "max_entry_bytes": 33554432,
      "max_entries": 2048,
      "directory": null
    }
  }
}
```

示例是 512 MiB 的缓存文件预算、32 MiB 的单条数据上限和 2048 条上限，**不是预检自动推荐或写入的生产配置**。
`max_entry_bytes` 不得超过 `max_disk_bytes`。配置错误在启动校验时拒绝。
缓存位于 `run_id=…/.work/.input-cache/`；诊断位于本次 invocation 的 `.input-cache/`。
若结果目录位于慢盘/挂载盘，可显式将 `directory` 设置为本机 NVMe 上的绝对路径，缓存会在该目录下
按 run 工作目录哈希隔离；预算是每个 run 的预算，退出清理仍有效，不改动其它 run 的缓存。
本版按 Linux/WSL 本地文件锁设计，不支持将它当 NFS/分布式缓存使用。

## 哪些数据共享，哪些不共享

| 数据 | 实现与边界 |
| --- | --- |
| 原视频、准备 clip | 原视频不修改；继续使用现有 FFmpeg 准备路径。准备 clip 的 SHA-256、大小和修改时间绑定缓存身份 |
| 采样帧 | 原尺寸 OpenCV BGR uint8；只有模型真正请求的帧才解码/缓存，不预先展开整段视频 |
| 帧共享范围 | 同一准备片段内相同帧号，HaMeR/FP/Fast3R 共享；键包含准备内容、片段身份和 OpenCV 构建身份 |
| HaMeR crop | 缓存官方 `ViTDetDataset.__getitem__` 生成的完整数组/标量，保持 resize、翻转、抗混叠、归一化、相机元数据和 batch 顺序 |
| crop 身份 | 图像内容、框中心/尺度、左右手、完整模型 crop 配置、dataset 源码与依赖版本；同一框也不能跨不同归一化配方错误命中 |
| FP | 共享 RGB 采样帧；保留按帧复用 RGB-D、对象独立 mesh 副本、独立 tracker；深度输入不放进此共享缓存 |
| Fast3R | 共享解码帧；保留 JPEG 和官方 loader，避免未经对照改变像素/预处理语义 |
| SAM2 | 暂保留官方 loader 专用解码路径，未实现与其它模型共用视频状态 |

本版没有跨重叠 clip 的源帧去重，也没有 CPU 预取池或跨帧 HaMeR batch。
共享帧要求解码器构建身份相同；不同 OpenCV/FFmpeg 构建使用不同键，避免把潜在像素差异静默混在一起。
本机 HaMeR/Fast3R 环境都是 OpenCV 4.11；FoundationPose 是 4.14，因而它能复用自身重复调用的帧，
但不会命中前两者生成的帧。若要在这些异构环境间也只解码一次，需要先统一并验证解码环境，
或把解码移到独立 CPU 输入准备边界；本轮未自动更改模型环境中的 OpenCV。
crop 当前主要用于同一输入重复诊断、重试或同 run 再消费；普通流水线对一个手框只消费一次时，
缓存写入可能只有成本而没有命中收益。是否启用应由代表性负载决定。

## memmap 生命周期与预算

缓存使用不可变 `.dat` 数组数据和含 SHA-256 的 JSON 元数据。读取时打开只读 memmap，校验内容，
复制成调用方可修改的普通数组，然后在 `finally` 中关闭映射。**没有 memmap 句柄或裸视图逃逸到模型、Ray 或 Writer**。
这样 FP/第三方预处理修改自己的数组时不会污染其它消费者；它也意味着本版不是零拷贝设计。

并发和异常规则：

1. 固定 64 个分片文件锁合并同键的并发 miss；计算 crop/解码在全局锁之外执行。
2. 全局文件锁保护读回、校验、复制、关闭以及发布/淘汰。读者映射仍打开时，淘汰者不能删除其文件。
3. 发布前同时检查整个 run 的缓存文件总字节和条目数，按最近使用时间淘汰；再以临时文件和原子 rename 发布。
4. 单条超限或连元数据都容纳不下时，直接返回本次计算结果并记录 bypass，不改变帧数、精度或 batch。
5. SHA、元数据、大小不匹配视为缓存损坏，删除后从原输入重建；缓存不是可信的阶段提交。
6. 内核在进程退出/被杀时释放文件锁；下一次打开缓存会回收未发布临时文件和孤立数据文件。
7. 片段结束（包括失败收尾）清除该片段的缓存；driver 关闭 workers 后清理剩余缓存。`keep_prepared` 不保留输入缓存。
8. 预检完成、报错或可捕获中断后先关闭 runner，再清理缓存数据，保留请求、报告、阶段结果和配置快照。
   协调进程被 SIGKILL/掉电时不能保证即时清理，遗留内容仍受同一预算约束并在下次访问时回收/淘汰。

`max_disk_bytes` 约束缓存数组和条目 JSON 的文件长度总和，包含发布中的数据；固定锁文件和策略文件不计入。
它不是文件系统块分配、整个临时盘或磁盘剩余容量的保证。原视频、准备 clip、深度副本、Fast3R JPEG、
结果文件、模型 debug 文件均有各自生命周期，不应假定受此预算限制。

`max_entry_bytes` 约束可缓存/映射的单条数据；每个 worker 同时至多在缓存读回区映射一个条目。
复制出的帧、DataLoader batch、模型私有堆、操作系统页缓存和 CUDA 显存**不受它硬限制**。
没有新增常驻 Python 数组 LRU；实际主机余量仍须结合进程树 RSS、并发 worker 数和原有反压预算判断。
这完成了有界输入缓存的机制，没有完成第五部分 E 的全进程 RAM/全临时盘统一准入。

## 可观测性

每个阶段提交及诊断逐次记录包含：

- `input_cache_hits/misses`，以及 `input_cache_frame_*` / `input_cache_crop_*` 分项。
- `input_cache_bypasses/corruptions/evictions`。
- `input_cache_peak_disk_bytes`：该阶段观测到的整个 run 缓存文件高水位；跨阶段汇总取最大，不相加。
- `input_cache_peak_mapped_bytes`：该进程缓存读回区的映射峰值；不是 RSS。
- `input_cache_read_seconds/build_seconds/write_seconds`：缓存读回（含校验和复制）、miss 计算及写入时间。
  这些数字不包括所有模型预处理工作，不能相减得到纯 GPU 时间。
- 诊断 `released_input_cache_bytes`：退出时实际删除的缓存数据/元数据总字节。

总计数/时间随阶段汇总累加，峰值取最大。缓存关闭时原有执行路径不改变。

## 验证方法

```bash
pytest -q egoworld/tests/test_input_cache.py egoworld/tests/test_diagnostics.py
```

测试覆盖跨进程同键只构建一次、并发共享预算、LRU 淘汰、超限旁路、帧字节一致、crop 键失效、
读取后无残留 memmap、消费者修改隔离、缓存损坏重建、进程被杀后的锁释放/临时文件回收、
以及 runner 成功/失败后的统计与清理。

真实模型对照使用同一视频、上游提交、模型参数与输出要求，分别启用/关闭缓存；
报告同时检查覆盖帧/行键、输出数组和数值差异。FP/PnP 等路径的随机性需要和重复基线一起看，
不能为获得“相等”静默放宽质量门槛。实测记录见 [回放与预检](replay.md)。
