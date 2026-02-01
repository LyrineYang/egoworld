文件描述： 当前的开发焦点、正在进行的任务与短期计划。

内容：

当前状态: 架构设计完成，进入 MVP (Minimum Viable Pipeline) 原型开发阶段。

正在进行:

搭建基于 Ray 的基础 Actor Pool 框架。

实现核心 ModelWrapper 基类。

跑通 "Scene Detect -> SAM 3 -> 3D Lifting" 的单条数据链路。

近期决策: 确定了 Parquet 表的数据 Schema（列定义），明确了不同 Actor 间的数据交换格式。

下一步计划:

集成 Hamer 和 DexRetargeting 模块。

在小规模数据集（100GB）上进行全链路压力测试。

部署 Prometheus 监控探针。