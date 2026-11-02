# 集成验收与 main 发布基线

2026-11-02：`codex/data-quality-efficiency` 已包含清洗/产品代码、main 的英文文档和网站提交 `529a175`，以及关联任务 `codex/stage-replay-gpu-probe` 至 `3e2aa5d` 的全部提交。验收基线为 `248b0c2`，工作区干净；远端 main 核验为 `529a175`，是集成分支的祖先，可快进合并。用户本次已授权合入 main、commit 和推送。

此前 [准备时的哈希清单](benchmarks/integration-readiness-2026-11-02.json) 是另一任务开发期间的历史观察；其中“主工作区仍未提交”的状态已经结束，不能用旧快照覆盖当前实现。

## 纳入的实现

- 模型前视频清洗、质量/运动统计、精确重复与空白过滤、按输出需求的分支准入。
- 13 类输出 recipes、验证后的阶段与派生缓存、哈希会话、可选可信资产登记、缓存管理及产品 CLI。
- 共同帧/crop 缓存、mask 搬运复用、HaMeR 合批、CPU/GPU 阶段拆分、runner 驻留、成本调度、回放和 GPU 预检。
- 手部坐标与 Dex 严格限位修正、左右手复现、RGB-D 标定世界接口。
- 独立的同输入/输出实验协议、真实 Data-Juicer CPU 清洗 pilot，以及共同 runner 的可选生产阶段 Mapper。

## 集成验收

| 检查 | 结果及边界 |
| --- | --- |
| 完整适用 CPU/Ray 回归 | 297 passed / 11 deselected，121.83 秒；native smoke 单独记录 |
| 当前生产源码 | `0b05f2ca23124447a18a0d3265f825def23d9b447be55c7f652f9030bb818dbc`，76 个 Python 文件与原生验证快照一致 |
| 原生手部/机器人 | 双手 12 行、右手机器人 6 行；与原样例全部数值列最大差为 0 |
| 左手 CPU-only | 只运行剩余 Dex，6 行左手机器人映射；左右手限位和独立 FK 通过 |
| 原生组合路径 | SAM/HaMeR/Fast3R/Dex 全部重算，观察到真实共同帧命中 |
| 世界与缓存 | RGB-D 样例 3 相机/3 物体/5459 世界点；最终源码全命中和派生复用通过 |
| 清洗 | 有效/重复/损坏/空白混合只保留有效输入；全坏输入在探测模型前返回 rejected |
| 分支过滤 | 显式过滤 HaMeR/Dex 时保留 boxes，不加载被过滤模型 |
| 结束状态 | 临时块数组零残留、模型 worker 零残留；handoff 与已下载模型保留 |

原始验收目录：`/mnt/d/EgoWorld/runs/implementation-20261102/`，包括 `completion-audit.json`、`final-regression.xml`、`integrated-native/acceptance.json` 和最终源码运行记录。上述数值属于短样例复现；世界误差与随机性边界见 [实现台账](efficiency-implementation-20261102.md)。这次文档和独立比较准备入口的变更没有改动已验收的生产包。

本轮新增比较准备：真实 Data-Juicer 左右手 Dex 阶段均通过，每侧 6 行完整字段精确一致；31 项针对性测试通过。详见 [比较接入验收](benchmarks/comparison-prep-2026-11-02.json)。未修改生产包，也未把这一阶段检查写成整链性能结论。

## 发布与比较

以正常 fast-forward 推进 main，保留两条分支历史，不 force push。推送后再核对远端 SHA 和本地干净状态；实际发布 SHA 由本次发布记录和 Git 历史确认。旧 Codex 检查点引用损坏不影响已验收源码；必要时使用独立传输仓库发布，保留原引用。

项目核心结果将以相同输入、模型、完整输出与质量规则下的有效产出和资源成本为准。CPU pilot 只证明实验接口和审计可用；完整产品任务的外部对照尚未完成。接下来按 [比较推进计划](comparison-next.md) 执行，不能把缓存冷热差异、少采样或 CPU pilot 启动差异算作框架优越性。
