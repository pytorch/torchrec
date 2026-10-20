# TorchRec

[English](README.MD) • [简体中文](README.zh.md)

**TorchRec** 是 PyTorch 专为大规模推荐系统（RecSys）构建的核心领域库，旨在提供推荐系统所需的通用稀疏性（sparsity）与并行性（parallelism）原语。TorchRec 支持将超大规模 Embedding 表切分并分片（sharded）至众多 GPU 上进行分布式训练与推理，**为 Meta 旗下的众多生产级推荐系统模型提供核心驱动**。

## 业界应用与实践

TorchRec 已广泛用于加速推荐系统的前沿突破与工程落地，部分代表性案例包括：
* [Meta 最新版 DLRM（深度学习推荐模型，Deep Learning Recommendation Model）](https://github.com/facebookresearch/dlrm) 基于 TorchRec 构建
* 论文 [Disaggregated Multi-Tower: Topology-aware Modeling Technique for Efficient Large-Scale Recommendation](https://arxiv.org/abs/2403.00877)
* 来自 Twitter 的开源推荐算法 [The Algorithm ML](https://github.com/twitter/the-algorithm-ml)
* [使用 Databricks 训练推荐模型 (Training Recommendation Models with Databricks)](https://docs.databricks.com/en/machine-learning/train-recommender-models.html)
* 论文 [Toward 100TB model with Embedding Offloading Paper](https://dl.acm.org/doi/10.1145/3640457.3688037)

## 快速入门

若要开始了解与学习 TorchRec，请参阅：
* 完整的 [TorchRec 官方教程 (TorchRec Tutorial)](https://pytorch.org/tutorials/intermediate/torchrec_intro_tutorial.html)
* [TorchRec 官方文档 (TorchRec documentation)](https://pytorch.org/torchrec/)：包含 TorchRec 全景架构概览与 API 参考

### 核心特性 (TorchRec Features)
- **并行原语 (Parallelism primitives)**：提供高效的并行原语，支持通过数据并行与模型并行混合模式（Hybrid DP/MP），轻松构建大规模、高性能的多卡/多节点模型。
- **分片器 (Sharders)**：支持多种 Embedding 表分片策略，包括数据并行（data-parallel）、按表分片（table-wise）、按行分片（row-wise）、表行混合分片（table-wise-row-wise）、按列分片（column-wise）以及表列混合分片（table-wise-column-wise）。
- **规划器 (Planner)**：能够针对具体模型自动生成全局最优的分片方案（Sharding plans）。
- **流水线训练 (Pipelined training)**：深度重叠（Overlap）数据加载设备传输（拷贝至 GPU）、卡间通信（`input_dist`）以及核心计算（前向与反向传播），实现极致性能提升。
- **高性能算子 (Optimized kernels)**：由 [FBGEMM](https://github.com/pytorch/FBGEMM/tree/main) 强力驱动，针对推荐系统进行了极致性能优化。
- **量化支持 (Quantization)**：支持低精度训练与推理，并支持将 TorchRec 模型针对 C++ 推理环境进行深度优化。
- **通用模块 (Common modules)**：内置推荐系统所需的通用模块库。
- **开源数据集 (RecSys datasets)**：提供常用的推荐系统数据集（Criteo 点击日志数据以及 MovieLens）。
- **端到端训练示例 (End-to-End examples)**：提供完整的端到端训练示例，例如在 Criteo 点击日志数据集上训练 DLRM 事件预测模型。

## 安装指南

请参阅官方文档中的 [快速入门 (Getting Started)](https://pytorch.org/torchrec/setup-torchrec.html) 章节，获取推荐的 TorchRec 安装配置方式。

### 源码编译安装 (From Source)

**通常情况下，无需从源码编译**。对于绝大多数使用场景，只需按照上述文档章节安装预构建版本即可。如果您希望从源码编译以体验最新功能改动，请执行以下步骤：

1. 安装 PyTorch。请参阅 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)。
   ```shell
   CUDA 12.6

   pip install torch --index-url https://download.pytorch.org/whl/nightly/cu126

   CUDA 13.0

   pip install torch --index-url https://download.pytorch.org/whl/nightly/cu130

   CPU

   pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

2. 克隆 TorchRec 仓库。
   ```shell
   git clone --recursive https://github.com/meta-pytorch/torchrec
   cd torchrec
   ```

3. 安装 FBGEMM。
   ```shell
   CUDA 12.6

   pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cu126

   CUDA 13.0

   pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cu130

   CPU

   pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

4. 安装其他依赖。
   ```shell
   pip install -r requirements.txt
   ```

5. 安装 TorchRec。
   ```shell
   python setup.py install develop
   ```

6. 验证安装。
   ```shell
   GPU 模式

   torchx run -s local_cwd dist.ddp -j 1x2 --gpu 2 --script test_installation.py

   CPU 模式

   torchx run -s local_cwd dist.ddp -j 1x2 --script test_installation.py -- --cpu_only
   ```
   有关启动分布式与远程作业的更多信息，请参阅 [TorchX](https://pytorch.org/torchx/)。

7. 如果您希望运行更复杂的示例，请查阅 TorchRec [DLRM 示例](https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/dlrm_main.py)。

## 贡献指南

有关为 TorchRec 贡献代码的详细信息，请参阅 [CONTRIBUTING.md](https://github.com/meta-pytorch/torchrec/blob/main/CONTRIBUTING.md)！

## 引用 (Citation)

如果您在研究或工程中使用了 TorchRec，请参考以下 BibTeX 条目引用本项目：
```bibtex
@inproceedings{10.1145/3523227.3547387,
author = {Ivchenko, Dmytro and Van Der Staay, Dennis and Taylor, Colin and Liu, Xing and Feng, Will and Kindi, Rahul and Sudarshan, Anirudh and Sefati, Shahin},
title = {TorchRec: a PyTorch Domain Library for Recommendation Systems},
year = {2022},
isbn = {9781450392785},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3523227.3547387},
doi = {10.1145/3523227.3547387},
abstract = {Recommendation Systems (RecSys) comprise a large footprint of production-deployed AI today. The neural network-based recommender systems differ from deep learning models in other domains in using high-cardinality categorical sparse features that require large embedding tables to be trained. In this talk we introduce TorchRec, a PyTorch domain library for Recommendation Systems. This new library provides common sparsity and parallelism primitives, enabling researchers to build state-of-the-art personalization models and deploy them in production. In this talk we cover the building blocks of the TorchRec library including modeling primitives such as embedding bags and jagged tensors, optimized recommender system kernels powered by FBGEMM, a flexible sharder that supports a veriety of strategies for partitioning embedding tables, a planner that automatically generates optimized and performant sharding plans, support for GPU inference and common modeling modules for building recommender system models. TorchRec library is currently used to train large-scale recommender models at Meta. We will present how TorchRec helped Meta’s recommender system platform to transition from CPU asynchronous training to accelerator-based full-sync training.},
booktitle = {Proceedings of the 16th ACM Conference on Recommender Systems},
pages = {482–483},
numpages = {2},
keywords = {information retrieval, recommender systems},
location = {Seattle, WA, USA},
series = {RecSys '22}
}
```

## 开源许可证

TorchRec 遵循 BSD 开源许可证，详情参见 [LICENSE](LICENSE) 文件。

---

> **免责声明 / Disclaimer**：本项目由开源社区志愿者以非官方身份发起并独立维护。若翻译内容与英文原版存在任何出入，请以英文官方文档为准。欢迎提交 PR 协同完善！
