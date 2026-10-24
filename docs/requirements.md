# 目标与需求

目标：完成单服务器离线第一视角视频处理流水线，所有分支先推进到可测试，再逐阶段实测、修复。
近期交付为 CLI、完整结构化输出、并发调度和可靠恢复；Demo 暂缓。

当前范围已确认：交付 SAM2/GroundingDINO、HaMeR、FoundationPose（有已有 RGB-D/网格时）、DexRetarget、Fast3R。
模型与输出要求见 [输出清单](outputs.md)。硬件传感器、力/触觉、复杂新模型不实现，Demo 暂缓。
命名弯曲角、世界坐标融合、动作/接触标签不纳入本轮启动验收。
开发历史与证据见 [项目复盘](overview.md)；高吞吐建议见 [调度设计](design.md)。

## 需求与实现对应

表中的测试证明相应代码行为；模型替身不证明真实推理、精度或吞吐量。

| ID | 当前要求 | 实现位置（相对 egoworld/src/egoworld） | 可执行测试（egoworld/tests） |
| --- | --- | --- | --- |
| R01 | 确定性 ID、帧对齐、片段长度上限与重叠 | manifests/build_manifest.py | test_manifest_build.py、test_cpu_readiness.py |
| R02 | 按依赖就绪调度；CPU/GPU/写盘重叠；限制活动片段与在途结果 | pipeline/engine.py、backend.py、stages.py | test_pipeline_logic.py、test_ray_integration.py |
| R03 | SAM2/GroundingDINO 输出 mask、实例和二维轨迹 | operators/sam2_op.py、groundingdino_op.py | test_sam2_logic.py、test_groundingdino_logic.py、test_native_models.py |
| R04 | HaMeR 输出 21 关节、MANO、左右手和相机平移 | operators/hamer_op.py | test_model_adapters.py、test_native_models.py |
| R05 | FoundationPose 从对齐 RGB-D、内参、网格得到物体到相机变换 | operators/foundationpose_op.py | test_model_adapters.py、test_stage_contracts.py、test_native_models.py |
| R06 | DexRetarget 按片段/手重置优化器，输出指定机器人关节序列 | operators/dex_retarget_op.py | test_model_adapters.py、test_native_models.py |
| R07 | Fast3R 输出相机、内参和有界点云；明确相对尺度与失败 PnP | operators/fast3r_op.py | test_model_adapters.py、test_stage_contracts.py、test_native_models.py |
| R08 | 阶段恢复、失败隔离、非阻塞重试、超时、worker 替换、损坏恢复 | pipeline/engine.py、backend.py、artifacts.py、state_store.py | test_pipeline_logic.py、test_ray_integration.py |
| R09 | 严格 schema、单位/参考系、代码/清单/本地模型身份；拒绝不支持配置 | contracts.py、config.py、utils/provenance.py | test_stage_contracts.py、test_operator_config_contract.py |
| R10 | 事件、阶段耗时、QC、汇总及可重复测试入口 | acceptance.py、cli.py、pipeline/artifacts.py | test_pipeline_logic.py、test_native_models.py、test_ray_integration.py |
| R11 | 用户按服务器文档配置；环境/资产预检、目录启动、清单复用与恢复；本地执行链路测试 | launch.py、preflight.py、worker.py、config.py | test_server_startup.py |

以上均已纳入当前实现范围，不采用“上一模型验收后才开发下一模型”的流程。
需求变化必须落到本表对应的代码与行为测试；字段类型只由 `contracts.py` 定义，`egoworld schema` 查看。
当前配置图由 `egoworld plan` 查看。CPU CI 已配置为运行契约、调度与真实 Ray 替身测试；托管 CI 实际结果另行记录。

## 验证边界

- 无 GPU：验证输入、五分支数据流、官方 API 形态的模型替身、真实 FFmpeg/Parquet、Ray 调度与恢复。
- 单卡待测：真实依赖、权重、输入资产、推理输出、显存、重复运行恢复；逐阶段测试并继续修复其余分支。
- 服务器待测：实际数据吞吐、CPU/磁盘瓶颈、长时间运行和真实 CUDA 故障恢复。
- 原设计的 GPU 利用率 >60%、失败率 <1% 是待测指标，不是当前事实。Prometheus/DCGM 自动监控尚未接线。

## 必需输入与范围

输入为本地恒定帧率视频。FoundationPose 另需按源帧编号对齐的深度序列、相机内参与带尺度网格；
DexRetarget 另需机器人 YAML/URDF、目标左右手、坐标旋转和明确关节单位。缺少资产时相关阶段明确失败。
不把未标定的 Fast3R 重建当成米制深度。

近期交付标准：[服务器操作说明](server.md)中的配置和命令有对应代码；本地启动/契约/调度测试通过；
缺依赖/资产时明确失败；服务器执行真实模型 smoke，发现的实际兼容性问题继续修复。
本地结果不用于承诺任何未经服务器验证的 CUDA/显存/吞吐表现。

不做：Demo UI、策略训练、多机存储调度、Kafka/Redis、exactly-once、跨片段全局 ID、机器人硬件控制。
近期包装定位：**面向大规模数据集的可恢复手—物体—场景结构化处理框架**。
可以描述扩展机制和设计目标；在实测前不写“已验证 TB 级吞吐”或“大规模处理准确率”。

## 接下来的产品开发顺序

保持一个 CLI、一套配置和一套阶段输出。当前修复先进入可复验版本，后续按下面顺序交付；
各模型仍可并行定位问题，不将上一分支验收作为开发下一分支的前置条件。

| 优先级 | 交付结果 | 完成条件 |
| --- | --- | --- |
| 现在 | 可靠的启动、停止和恢复 | 五项审查问题有反例回归；CPU/Ray 通过；用户按 server.md 可配置并进入真模型试跑 |
| 下一步：单卡可用 | 固定真实模型环境与少量可复验样本 | 每个启用阶段的短样本、完整单卡、一次中断恢复；保留命令、原始输出、模型版本与环境导出；抽查几何和左右手 |
| 随后：更容易使用 | 针对首次真实试用中遇到的配置/错误逐项简化 | 同一份操作说明能在干净服务器重走；错误能指出缺什么、去哪里修；输出可被独立读取器读取 |
| 再后：提高吞吐 | 根据瓶颈选择一次小优化 | 固定相同输入/采样/输出，拆分加载、解码、推理、后处理、写入计时并记录 RAM/VRAM；先优化最贵一项 |

吞吐候选顺序为：读取所需列和复用同帧 RGB-D → 限制库线程与临时盘 → 按模型批次摊薄加载。
是否采用共享帧缓存或 Ray Data，由小规模对照决定，不同时维护两套生产后端。
具备真实小样本与恢复证据后再考虑发布可用版本；Demo、新模型和多机扩展继续后置。

## 产品可用性复核：16GB 单卡

当前定位为**可进入服务器试用的开发版 CLI**。安装、配置、预检、短样本、处理、恢复已有对应代码和本地测试；
仍不能以“完整产品已验证”对外承诺无人值守成功或五个真实模型均可用。

本轮先解决直接阻塞：Fast3R 上游的 device/返回值/precision 调用不一致、原生缓存与分环境模型显存叠加，
并增加 smoke --profile 16gb。该配置缩小工作量、隔离 GPU 阶段，不包含重复读取或吞吐重构。

试用剩余门槛：

1. 用真实 16GB GPU 型号确认 BF16 能力，再检查各原生环境和资产；尤其 HaMeR/MANO、机器人 YAML/URDF 仍需用户准备。
2. 保存一次启用阶段完整 smoke 报告，并人工检查左右手/对象、姿态和点云；仅输出非空不能证明含义正确。
3. 在真实数据上处理一次目录、停止并恢复，保留日志与环境版本。缺少某分支资产时明确关闭，报告注明实际覆盖范围。

目前最影响首次使用的是依赖/资产准备和真模型兼容性。先收集这轮试用中的失败并修复，效率优化排在其后。
