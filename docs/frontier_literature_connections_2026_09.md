# ⚡ Router-Tuning (Mixture-of-Depths): 每日前沿文献关联与动态深度/Token 路由落地库 (2026-09 — 2026-10)

**Document ID:** `ROUTERTUNING-LIT-202609` | **Last Updated:** `2026-10-01` | **Target Path:** `docs/frontier_literature_connections_2026_09.md` | **Total Routed Papers:** `23`

> [!IMPORTANT]
> **🔗 跨仓库文献引用链闭环 (Cross-Repository Reference Chain Closure)**
> 本文件由每日 AI 前沿论文精读流水线自动路由生成，专门收录与我们 **EMNLP 2025 代表作 (*Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers*, `CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths`)** 直接关联的动态 Token 级层跳过（Mixture-of-Depths）、自适应层选择（`ASL`, `CLSE`, `DEE-VLA`, `DORA`, `FocusVLA`, `EvoESAP`）、层内可逆稀疏路由（`Token Sparse Attention`）、低秩 Lipschitz 路由器（`L2R`）、双阈值自适应激活（`CARE`）以及循环动态停止（`DeepLoop`, `Training-Free Looped`）最新 arXiv 论文笔记。
> 每一篇收录文献均包含：**核心痛点、底层数学公式、ASCII 架构图、关键实测指标**，以及**与 `Router-Tuning-Mixture-of-Depths` 仓库具体代码模块和我们已发表代表作（Our Works）的双向锚定**。

---

## 🌟 1. 核心关联文献与本仓库模块映射速查表 (Executive Reference-to-Module Matrix)

| 收录日期 | 论文标题与 arXiv 链接 | 关键实测收益 / 核心结论 | 锚定本仓库代码模块与文档路径 (`Target Module`) | 原始精读归档 |
| :---: | :--- | :--- | :--- | :---: |
| `2026-10-01` | [**AIMER & EvoESAP**](https://arxiv.org/abs/2603.18492) (`arXiv:2603.18492`) | **`AIMER` 超越基于 C4 校准集的强基线且速度快几个数量级**：在涵盖 `7B` 至 `47B` 不同架构的 MoE 语言模型及 **16 个多样化基准**上，免校准的 `AIMER` 不仅全面超越现有免校准方法，更在跨... | `router_tuning/` (Evolutionary Non-Uniform Layerwise Token/Expert Capacity Allocation) | [2026-10-01](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-10-01_ai_paper_notes.md) |
| `2026-10-01` | [**FocusVLA & Navigation Heads**](https://arxiv.org/abs/2603.28740) (`arXiv:2603.28740`) | **`FocusVLA` 提升精细操作与收敛速度**：在仿真与真实世界机器人基准上，`FocusVLA` 通过切断非视觉捷径并显式抑制无关背景噪声，在灵巧操作任务上大幅提升任务成功率并显著加快训练收敛速度。 | `router_tuning/` (Action-Conditioned Query-Guided Visual Token Routing Gate) | [2026-10-01](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-10-01_ai_paper_notes.md) |
| `2026-09-30` | [**ACPruner & SCOPD**](https://arxiv.org/abs/2609.34558) (`arXiv:2609.34558`) | `ACPruner` (`2609.34558`) 保留 64/576 视觉 Token 维持 97.4% 精度；`SCOPD` (`2609.34044`) 10% 视觉 Token 保留率下 13 基准保留率：Vanilla 86.37%、SCOPD 90.49%、SCOPD+ 92.43% | `router_tuning/` (Biased Attention Coverage Maximization for Token Routing) | [2026-09-30](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-30_ai_paper_notes.md) |
| `2026-09-30` | [**Dynamic Flow, Static Graph & DORA**](https://arxiv.org/abs/2609.34727) (`arXiv:2609.34727`) | **端侧静态图 NPU 首字延迟骤降**：在高通骁龙 8 Elite（Hexagon NPU）与端侧 SoC 上运行 `Qwen2.5-3B/7B` 与 `Llama-3.2-3B`... | `router_tuning/` (Dynamic Online RL Advantage-Guided Token Routing) | [2026-09-30](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-30_ai_paper_notes.md) |
| `2026-09-29` | [**✂️ CoverPruner & SFPruner**](https://arxiv.org/abs/2609.03158) (`arXiv:2609.03158`) | 在 LLaVA-NeXT、Qwen2.5-VL 与 InternVL-2.5 等高分辨率多模态模型上，当剪除 **80%–88.9% 视觉 Token**（仅保留 64–128 个 Token）时，`CoverPruner` 与... | `router_tuning/` (Coverage-Preserving Token Routing & Barycentric Aggregation) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-29` | [**🦾 DEE-VLA**](https://arxiv.org/abs/2609.29382) (`arXiv:2609.29382`) | 在 LIBERO（Spatial / Object / Goal / Long）与真机双臂灵巧操作任务上，`DEE-VLA` 在成功率与全深度 10-NFE 基线持平（甚至因减少自由空间过拟合而提升 **+0.8%**）的同时，平... | `router_tuning/` (Decoupled Early-Exit Halting Gates across Multimodal & Action Towers) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-28` | [**✂️ CLSE**](https://arxiv.org/abs/2606.24165) (`arXiv:2606.24165`) | 在 LLaVA-NeXT-7B/13B 与 Qwen2.5-VL-7B 上，免训练剪除 **75% 视觉 Token** 时，在 DocVQA、ChartQA、TextVQA 与 Video-MME 等 10 项多模态基准上平均保... | `router_tuning/` (Spectral Entropy Inflection Layer Allocation for Token Routing) | [2026-09-28](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-28_ai_paper_notes.md) |
| `2026-09-28` | [**✂️ ASL**](https://arxiv.org/abs/2601.07667) (`arXiv:2601.07667`) | 在 Llama-3.1-8B/70B 与 Qwen2.5-14B 上，针对 RULER、InfiniteBench 与 Needle-in-a-Haystack（128K 上下文）评测表明：在相同的... | `router_tuning/` (Marginal Information Gain Schedule for Dynamic-Depth MoD Layers) | [2026-09-28](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-28_ai_paper_notes.md) |
| `2026-09-27` | [**L2R**](https://arxiv.org/abs/2601.21349) (`arXiv:2601.21349`) | **语言与视觉双模态全面验证**：在基于 **OLMoE** 的语言模型预训练/微调以及 **ImageNet** 视觉 MoE 骨干网络上，L2R 将路由器参数量削减 **60%–75%**，同时在相同激活专家预算下将下游任务困... | `router_tuning/` (Low-Rank + SIPS Lipschitz-Stabilized MoD Router Head) | [2026-09-27](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-27_ai_paper_notes.md) |
| `2026-09-26` | [**🔄 LoopMoE**](https://arxiv.org/abs/2606.04438) (`arXiv:2606.04438`) | **等参数量与等 FLOPs 双向碾压**：在语言建模基准与常识推理任务上，循环 $K=2\sim 4$ 步的 `LoopMoE` 在相同活跃参数量下显著优于标准稠密 Looped 模型，且在相同总参数预算下逼近非共享深层 MoE... | `router_tuning/` (Dynamic Token Halting across Looped Transformer Steps) | [2026-09-26](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-26_ai_paper_notes.md) |
| `2026-09-26` | [**🤖 VLA-Pruner**](https://arxiv.org/abs/2511.16449) (`arXiv:2511.16449`) | 在 OpenVLA 与主流机器人操控基准（LIBERO-Spatial / Object / Goal / Long）上，剔除 **50%–75% 视觉 Token** 仍保持与全量 Token 持平的任务成功率，端到端控制频率显... | `router_tuning/` (Dynamic Token Halting across Looped Transformer Steps) | [2026-09-26](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-26_ai_paper_notes.md) |
| `2026-09-25` | [**Fully Looped Transformer**](https://arxiv.org/abs/2605.18797) (`arXiv:2605.18797`) | 在完全不增加任何额外参数（0 Extra Parameters）的条件下，Fully Looped Transformer 在 $K=8, 12$ 步循环预训练中完全消除了传统 Looped Transformer 的梯度尖峰（G... | `router_tuning/` (Dynamic Token Halting across Looped Transformer Steps) | [2026-09-25](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-25_ai_paper_notes.md) |
| `2026-09-24` | [**LearnPruner**](https://arxiv.org/abs/2604.23950) (`arXiv:2604.23950`) | 在 **LLaVA-1.5/NeXT** 与 **Qwen2-VL** 上，LearnPruner 仅保留 **11.1%–16.7% 视觉 Token**，FLOPs 降低 **68%**，在 10 项多模态基准上的平均精度达到... | `router_tuning/` (`CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths`) | [2026-09-24](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-24_ai_paper_notes.md) |
| `2026-09-24` | [**Training-Free Looped Transformers**](https://arxiv.org/abs/2605.23872) (`arXiv:2605.23872`) | 在完全零训练（Zero Finetuning）的 **Llama-3-8B** 与 **Mistral-7B** 上，对中段 6 层额外循环 $K=2$ 次，在 GSM8K、ARC-Challenge 与逻辑推理任务上直接获得... | `router_tuning/` (Dynamic Token Halting across Looped Transformer Steps) | [2026-09-24](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-24_ai_paper_notes.md) |
| `2026-09-23` | [**StepKV**](https://arxiv.org/abs/2609.22158) (`arXiv:2609.22158`) | 在 **AIME 2025**、**MATH-500** 与 **GPQA-Diamond** 上，StepKV 在压缩 **65%–75% CoT KV 缓存** 的条件下，相比逐 Token 驱逐的 SnapKV / H2O... | `router_tuning/` (`CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths`) | [2026-09-23](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-23_ai_paper_notes.md) |
| `2026-09-23` | [**HetDPT**](https://arxiv.org/abs/2607.03784) (`arXiv:2607.03784`) | 在 **DeiT**、**Swin** 与 **CLIP-ViT-L/14** 上，HetDPT 在相同 **1.5x–1.8x 硬件实测加速比** 下，比整块深度剪枝提升了 **`+1.9%` 至 `+3.2%`** 的 Ima... | `router_tuning/` (Decoupled Attention-MoD vs FFN-MoD Routing Budgets) | [2026-09-23](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-23_ai_paper_notes.md) |
| `2026-09-23` | [**HiMoE-VLA**](https://arxiv.org/abs/2512.05693) (`arXiv:2512.05693`) | 在跨 50+ 任务的 Open-X Embodiment 与仿真套件上，HiMoE-VLA 比同激活参数量的稠密 VLA 与单层 MoE-VLA 平均成功率提升 **`+8.7%`**。 | `router_tuning/` (`CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths`) | [2026-09-23](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-23_ai_paper_notes.md) |
| `2026-09-23` | [**D-Cut**](https://arxiv.org/abs/2607.14647) (`arXiv:2607.14647`) | 在 Batch Size = 16–64 的生产级投机解码服务中，D-Cut 将验证阶段算力开销削减 **38%**，端到端吞吐在 EAGLE-2 基线上进一步提升 **1.42x**。 | `router_tuning/` (Confidence-Margin Early Depth Termination) | [2026-09-23](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-23_ai_paper_notes.md) |
| `2026-09-21` | [**DeepLoop**](https://arxiv.org/abs/2607.13491) (`arXiv:2607.13491`) | 在循环深度从 $K=2$ 扩展至 ** $K=16$ ** 的语言与数学推理预训练中，标准 Pre-LN 循环架构在 $K \ge 6$ 时完全发散，而 **DeepLoop** 稳定收敛并实现随循环次数 $K$ 对数线性下降的测... | `router_tuning/` (Dynamic Token Halting across Looped Transformer Steps) | [2026-09-21](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-21_ai_paper_notes.md) |
| `2026-09-21` | [**RotateK**](https://arxiv.org/abs/2605.19218) (`arXiv:2605.19218`) | 在 **LLaVA-NeXT**、**Qwen2-VL-7B** 与 **InternVL-2** 上，RotateK 剪除 **50%–60% 的 Key 通道**而无需微调，且与视觉 Token 剪枝（如 FastV / VL... | `router_tuning/` (`CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths`) | [2026-09-21](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-21_ai_paper_notes.md) |
| `2026-09-21` | [**Token Sparse Attention**](https://arxiv.org/abs/2602.03216) (`arXiv:2602.03216`) | 在 64K–128K 多跳检索与大海捞针基准（RULER Multi-Hop Tracing）上，不可逆 Token 剪枝在 70% 稀疏度下准确率跌至 `31.2%`，而 **Token Sparse Attention** 保... | `router_tuning/` (Reversible Token Bypass vs Irreversible Drop in MoD) | [2026-09-21](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-21_ai_paper_notes.md) |
| `2026-09-20` | [**CARE**](https://arxiv.org/abs/2607.26052) (`arXiv:2607.26052`) | 在多任务 MoE-LoRA 与稀疏 MoE 语言模型上，CARE 在削减 **32%–45% 平均专家激活 FLOPs** 的同时，在常识推理、代码与数学基准上全面持平甚至超越固定 Top- $k$ 基线（`+0.9%` 平均准确... | `router_tuning/` (Cumulative Probability Nucleus + Marginal Jump Dynamic Depth Gate) | [2026-09-20](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-20_ai_paper_notes.md) |
| `2026-09-18` | [**🧩 MoE-Tile**](https://arxiv.org/abs/2609.09112) (`arXiv:2609.09112`) | **硬件测试平台**：NVIDIA H100 80GB SXM5 与 B200 GPU 集群； | `router_tuning/` (`CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths`) | [2026-09-18](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-18_ai_paper_notes.md) |

---

## 🔎 2. 来源核验、推导边界与复现补充规范 (Source Verification & Reproducibility Notes)

### 🔎 来源核验与研究补充（2026-10-01）

本日共涵盖 **6 个主题组、12 篇 arXiv 论文**。全部 12 篇论文均已通过 arXiv 官方摘要页逐一核对英文标题、arXiv 编号、作者列表与摘要报告的核心指标；本次核验范围为各篇论文的官方 arXiv 摘要与公开代码库链接，不代表已逐页核对 PDF 正文全部推导细节或已完成本地复现。

**引用与原始指标核验说明**：
1. **具身与视觉 Token 剪枝组**：`IAprune`（[arXiv:2603.22991](https://arxiv.org/abs/2603.22991)）摘要报告在 4 种具身操作策略、3 个仿真基准与真机平台上评估，在 LIBERO 上匹配未剪枝策略精度并取得 **`1.54×` 加速**，在真机平台上达到 **`1.48×` 加速**；`Rényi Entropy (Col-Ln)`（[arXiv:2603.27900](https://arxiv.org/abs/2603.27900)）提出基于 Rényi 熵的免训练指标 `Col-Ln` 从首层识别高信息量视觉 Token。两篇论文的级联组合属于本仓库提出的下一步研究建议，非原论文联合实验。
2. **MoE 专家剪枝组**：`AIMER`（[arXiv:2603.18492](https://arxiv.org/abs/2603.18492)）与 `EvoESAP`（[arXiv:2603.06003](https://arxiv.org/abs/2603.06003)，开源代码 `https://github.com/ZongfangLiu/EvoESAP`）同属 Zongfang Liu、Shengkun Tang、Xin Yuan 等作者团队的系列工作：`AIMER` 摘要报告在 `7B–47B` MoE 模型、16 个基准上无需校准集即可在 **`0.22–2.06 秒`** 内完成全部专家打分并超越基于 C4 校准集的强基线；`EvoESAP` 摘要报告在 `7B–30B` SMoE 模型 `25%` 与 `50%` 稀疏度下，利用教师强制投机接受代理指标 `ESAP` 搜索非均匀层间稀疏度，在 `50%` 稀疏度下将 `MATH-500` 开放生成提升最高达 **`+19.6%`**。
3. **KV 缓存压缩组**：`MixedDimKV`（[arXiv:2603.20616](https://arxiv.org/abs/2603.20616)）摘要报告在 LongBench 上仅用 **`6.25%` KV 缓存**即取得与全注意力相当的性能，在 `50K` 上下文长度的大海捞针（NIAH）测试中仅用 **`0.26%` 缓存**保持 **`100%` 准确率**；`DapQ`（[arXiv:2603.11564](https://arxiv.org/abs/2603.11564)）摘要报告在 **`3%` KV 缓存预算**下于 NIAH 取得高达 **`99.5%` 的近无损准确率**。
4. **具身 VLA 视觉聚焦与异常检测组**：`FocusVLA`（[arXiv:2603.28740](https://arxiv.org/abs/2603.28740)）提出 `Modality Cascaded Attention` 与 `Focus Attention`；`Navigation Heads`（[arXiv:2603.13782](https://arxiv.org/abs/2603.13782)）摘要报告在冻结 VLA 超过一千个注意力头中，仅组合 **3 个导航头（Navigation Heads）** 即可实现 **`44.6%` 的路径偏离检测率**与 **`11.7%` 的低误报率**，并在检测到偏离时触发轻量 RL 策略执行最短路径回滚。
5. **流匹配耦合蒸馏与混合世界模型组**：`The Coupling Within (NFM)`（[arXiv:2603.09014](https://arxiv.org/abs/2603.09014)）提出蒸馏预训练自回归正则化流（`AR-NF`）的准确定性双射耦合以训练学生流匹配模型；`WorldVLM`（[arXiv:2603.14497](https://arxiv.org/abs/2603.14497)）将高层 VLM 行为指令生成与底层自动驾驶世界模型动态预测相结合。
6. **元认知自指进化与防课程坍塌组**：`Hyperagents`（[arXiv:2603.19461](https://arxiv.org/abs/2603.19461)，开源代码 `https://github.com/facebookresearch/Hyperagents`）提出 `DGM-Hyperagents (DGM-H)`；`Prism`（[arXiv:2603.13309](https://arxiv.org/abs/2603.13309)）摘要报告在 7 个数学推理基准中的 6 个取得最高准确率，在 AMC 上较 `R-Zero` 提升 **`+3.98` 分**、在 Minerva Math 上提升 **`+3.68` 分**，并构建了包含 **`100k` 道数学题的 `Prism-Math` 数据集**。

| 主题组 | 原始论文来源 |
| :--- | :--- |
| 具身与早期视觉 Token 剪枝 | [IAprune (`2603.22991`)](https://arxiv.org/abs/2603.22991)、[Rényi Entropy `Col-Ln` (`2603.27900`)](https://arxiv.org/abs/2603.27900) |
| MoE 免校准打分与非均匀剪枝 | [AIMER (`2603.18492`)](https://arxiv.org/abs/2603.18492)、[EvoESAP (`2603.06003`)](https://arxiv.org/abs/2603.06003) |
| 异构维度与位置伪查询 KV 压缩 | [MixedDimKV (`2603.20616`)](https://arxiv.org/abs/2603.20616)、[DapQ (`2603.11564`)](https://arxiv.org/abs/2603.11564) |
| 具身 VLA 视觉利用与内生异常检测 | [FocusVLA (`2603.28740`)](https://arxiv.org/abs/2603.28740)、[Navigation Heads (`2603.13782`)](https://arxiv.org/abs/2603.13782) |
| 正则化流耦合蒸馏与世界模型-VLM | [Normalized Flow Matching `NFM` (`2603.09014`)](https://arxiv.org/abs/2603.09014)、[WorldVLM (`2603.14497`)](https://arxiv.org/abs/2603.14497) |
| 元认知自指智能体与防课程坍塌 | [Hyperagents `DGM-H` (`2603.19461`)](https://arxiv.org/abs/2603.19461)、[Prism (`2603.13309`)](https://arxiv.org/abs/2603.13309) |

**推导与实现边界**：后文给出的统一数学形式旨在清晰呈现各方法的核心算子结构，具体超参数定义、归一化常数与子模块变体应以各论文 PDF 原文为准。例如，`Rényi Entropy (Col-Ln)` 的核矩阵构造与阶数 $\alpha$ 取值、`AIMER` 在不同 FFN 矩阵（`gate_proj` / `up_proj` / `down_proj`）上的聚合维度、`MixedDimKV` 在张量核心（Tensor Core）上的内存对齐开销，以及 `NFM` 中教师 `AR-NF` 逆映射采样成本，均需在复现时对照原论文核验。将同一主题组的两篇论文串联（如 `AIMER` 排序接入 `EvoESAP` 层间搜索）属于我们的跨论文融合设计，不应归因为原论文已报告结果。

**建议复现顺序**：（1）优先在 `OLMoE` / `Qwen3-MoE` 上直接运行开源的 `EvoESAP` 与免校准 `AIMER`（零训练成本，数秒内可验证层内排序与层间非均匀分配收益）；（2）在 LIBERO 闭环评测中测试免训练的 `IAprune` 边界残差修正在低保留率下的抓取成功率与 50 Hz 控制周期延迟；（3）在 LongBench 与 NIAH 上对比 `DapQ` 位置伪查询与 `MixedDimKV-H` 的显存-精度帕累托前沿；（4）在 `TraceCraft` 与 `stock_prediction` 的自进化循环中引入 `Prism` 的嵌入语义分区覆盖与 ZPD 难度门禁。详细实验建议见[同日新闻](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/news/2026-10-01_daily_news.md)。

### 🔎 来源核验与研究补充（2026-09-30）

本日实际为 6 个主题组、12 篇论文。本次核对标题与编号，不代表已核对全部公式、实验表或完成复现。

**引用纠正**：SCOPD 的正确编号为 [2609.34044](https://arxiv.org/abs/2609.34044)。原笔记中的 `2609.33918` 实际对应 *Green AI: Cost of LLM-Based Code Completion*，后文涉及 SCOPD 的该编号均以此更正为准。

**指标纠正**：SCOPD 摘要在 10% 视觉 Token 保留率、13 个基准下报告相对未剪枝模型的性能保留率：Vanilla 86.37%、SCOPD 90.49%、SCOPD+ 92.43%。后文“99.5% 恢复率”、5,000 条训练指令、1 Epoch、68% 延迟降低及 79% 缓存压缩未获本次核验支持，撤回这些具体数值。ACPruner 与 SCOPD 的组合应视为研究建议，不能当作论文已报告的联合实验。

| 主题组 | 原始论文来源 |
| :--- | :--- |
| 视觉剪枝与蒸馏 | [ACPruner](https://arxiv.org/abs/2609.34558)、[SCOPD](https://arxiv.org/abs/2609.34044) |
| MoE 服务 | [SlimWise](https://arxiv.org/abs/2609.34117)、[CascadeEP](https://arxiv.org/abs/2609.33252) |
| 静态图与动态剪枝 | [Dynamic Flow, Static Graph](https://arxiv.org/abs/2609.34727)、[DORA](https://arxiv.org/abs/2609.34325) |
| 流匹配 | [CAT-Flow](https://arxiv.org/abs/2609.01746)、[MSFM](https://arxiv.org/abs/2609.35454) |
| 具身与世界模型 | [VLaRL](https://arxiv.org/abs/2609.30868)、[Programmable World Model](https://arxiv.org/abs/2609.10540) |
| 自我改进智能体 | [AutoDataBench](https://arxiv.org/abs/2609.35025)、[SelfOp](https://arxiv.org/abs/2609.22792) |

**推导与实现边界**：后文 KL 公式的方向为教师到学生，不应称为学生到教师的反向 KL；隐状态对齐等组合设计仍需全文逐式核验。次模近似保证需核对非负、单调、归一化与基数约束；流形收缩结论需明确成立区域与扰动假设。跨仓映射表仅为候选适配位置，本次没有检查其他仓库路径或执行跨仓写入。

**建议复现顺序**：先分别复现 ACPruner、SCOPD，再测组合；随后验证 MoE 在长短混合请求下的质量与吞吐，最后测试固定 NFE 下的流匹配误差。记录论文版本、代码 commit、模型与数据版本、随机种子、硬件及预算；同时报告分任务性能、端到端延迟和峰值显存。智能体技能更新应使用独立保留任务，防止验证集泄漏。详细实验建议见[同日新闻](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/news/2026-09-30_daily_news.md)。


---

## 📐 3. 逐篇论文深度机制解构、数学公式与本仓库落地指南 (Per-Paper Deep-Dive Cards)

### 3.1 [2026-10-01] AIMER & EvoESAP: Calibration-Free Weight Concentration MoE Expert Pruning & Speculative-Acceptance Evolutionary Non-Uniform Allocation (`arXiv:2603.18492` & `arXiv:2603.06003`)
* **论文标题**：
  1. *AIMER: Calibration-Free Task-Agnostic MoE Expert Pruning* (`arXiv:2603.18492`)
  2. *EvoESAP: Non-Uniform Expert Pruning for Sparse MoE* (`arXiv:2603.06003`)
* **核心关键词**：`moe`, `expert pruning`, `aimer`, `evoesap`, `esap`, `calibration-free`, `non-uniform sparsity`, `speculative decoding`, `capacity-aware`, `reap`

#### 📌 核心痛点与研究动机 (Motivation & Pain Points)
稀疏 Mixture-of-Experts（SMoE）大模型的部署受限于全量专家池的显存占用。当前训练后专家剪枝（Post-Training Expert Pruning）存在两大核心痛点：
1. **层内排序对校准集高度敏感且预处理昂贵（`AIMER` 动机）**：以 `Frequency`、`EAN`、`SEER`、`REAP` 为代表的现有方法均依赖在特定校准集（如 C4）上跑前向传播以统计路由频率或专家激活范数。这不仅耗费大量 GPU 预处理时间，更严重的是，校准集的语料分布偏差会导致剪枝后的模型在代码、数学或跨语言任务上出现偏科退化。
2. **跨层默认均匀稀疏度破坏敏感层表达力（`EvoESAP` 动机）**：几乎所有现有专家剪枝方法默认在每一层剪掉相同比例（Uniform Sparsity）的专家。然而不同 MoE 层的功能冗余度差异极大；若想搜索最优的跨层非均匀稀疏度分配（Non-Uniform Allocation），在每个候选配置上跑完整的自回归长文本生成（如 `MATH-500`）评估将产生不可承受的指数级计算开销。

#### ⚙️ 核心机制与数学公式推导 (Core Mechanism & Mathematical Formulation)
**第一部分：`AIMER` 的免校准绝对均值/均方根比（Absolute Mean over RMS）专家权重集中度准则**  
`AIMER` 发现：经过充分预训练的 MoE 模型，功能独特且不可替代的“高价值专才专家”在权重分布上表现出特定的结构集中度模式，而冗余专家的权重分布则更为散乱或同质。设第 $\ell$ 层第 $e$ 个专家的权重矩阵为 $W _ {\ell, e} \in \mathbb{R}^{d _ {\text{out}} \times d _ {\text{in}}}$ （共含 $M = d _ {\text{out}} d _ {\text{in}}$ 个参数元素）。`AIMER` 定义无需任何激活输入、纯基于权重的**绝对均值与均方根之比（Absolute Mean over Root Mean Square）**重要性准则：

$$
\mathcal{S} _ {\text{AIMER}}\left(W _ {\ell, e}\right) = \frac{\mathrm{Mean}\left(|W _ {\ell, e}|\right)}{\mathrm{RMS}\left(W _ {\ell, e}\right)} = \frac{\frac{1}{M} \sum _ {u=1}^{d _ {\text{out}}} \sum _ {v=1}^{d _ {\text{in}}} \left| W _ {\ell, e}^{(u, v)} \right|}{\sqrt{\frac{1}{M} \sum _ {u=1}^{d _ {\text{out}}} \sum _ {v=1}^{d _ {\text{in}}} \left( W _ {\ell, e}^{(u, v)} \right)^2}} = \frac{\lVert \mathrm{vec}(W _ {\ell, e}) \rVert _ 1}{\sqrt{M} \cdot \lVert \mathrm{vec}(W _ {\ell, e}) \rVert _ 2} \in \left[\frac{1}{\sqrt{M}}, 1\right]
$$

该比值本质上是权重向量归一化后的 $\ell _ 1 / \ell _ 2$ 范数比，纯在 GPU 上做张量规约即可在**毫秒至秒级（`0.22–2.06s`）**完成百亿参数 MoE 全模型专家排序，彻底摆脱校准集偏差。

**第二部分：`EvoESAP` 的教师强制投机接受率代理（`ESAP`）与跨层非均匀演化搜索**  
为将专家剪枝解耦为**“固定层内排序 + 优化跨层预算分配 $\mathbf{k} = (k _ 1, \dots, k _ L)$ ”**（满足全局预算约束 $\sum _ {\ell=1}^L k _ \ell = K _ {\text{total}}$ ），`EvoESAP` 借鉴投机解码（Speculative Decoding）中的草稿接受率定理，提出无需自回归解码、仅需在教师轨迹 $y = (y _ 1, \dots, y _ T)$ 上做**单次并行教师强制（Teacher-Forced）前向传播**的 **`ESAP`（Expected Speculative Acceptance Proxy）**：

$$
\mathrm{ESAP}(\mathbf{k}) = \frac{1}{| \mathcal{D} _ {\text{val}} |} \sum _ {y \in \mathcal{D} _ {\text{val}}} \frac{1}{T} \sum _ {t=1}^T \min\left(1, \frac{p _ {\text{pruned}}\left(y _ t \mid y _ {<t}; \mathbf{k}\right)}{p _ {\text{full}}\left(y _ t \mid y _ {<t}\right)}\right) \in [0, 1]
$$

由于 $\mathrm{ESAP}(\mathbf{k})$ 有界、平滑且单次评估仅需一次并行 Prefill，`EvoESAP` 以 $\mathrm{ESAP}(\mathbf{k})$ 为适应度函数运行演化搜索（通过保持总预算不变的层间专家配额突变算子 $k _ a \leftarrow k _ a + \Delta, k _ b \leftarrow k _ b - \Delta$ ），可作为即插即用模块赋能 `AIMER`、`Frequency`、`EAN`、`SEER` 与 `REAP` 等任意层内排序准则。

#### 🎨 算法架构图与实现伪代码 (Architecture & Pseudocode)
```
====================================================================================================
   AIMER (秒级免校准权重集中度层内排序) + EvoESAP (投机接受率代理跨层非均匀演化搜索) (arXiv:2603.18492 & 06003)
====================================================================================================

  [Pretrained SMoE Model (7B ~ 47B, L Layers, E Experts/Layer)]
                  │
                  ▼
  (Step 1: Within-Layer Ranking — AIMER or REAP/SEER/EAN)
  • AIMER 免校准计算每层专家权重 |W|_1 / (sqrt(M) * ||W||_2)，仅需 0.22 ~ 2.06 秒完成全模型层内排序
                  │ (固定各层内部专家剔除先后顺序)
                  ▼
  (Step 2: Across-Layer Budget Allocation — EvoESAP Evolutionary Search)
  • 种群初始化: 生成满足 ∑ k_l = K_total 的候选非均匀层间预算向量 k = (k_1, ..., k_L)
  • 快速适应度评估 (Teacher-Forced ESAP):
    并行前向计算 E_t [ min(1, p_pruned(y_t | y_<t; k) / p_full(y_t | y_<t)) ] (零自回归生成开销!)
  • 演化交叉与配额转移突变 ──► 输出最优非均匀专家保留配置 k* (在 50% 稀疏度下 MATH-500 提升 +19.6%)
====================================================================================================
```

#### 📊 实验指标与核心结论 (Experimental Results & Key Takeaways)
* **`AIMER` 超越基于 C4 校准集的强基线且速度快几个数量级**：在涵盖 `7B` 至 `47B` 不同架构的 MoE 语言模型及 **16 个多样化基准**上，免校准的 `AIMER` 不仅全面超越现有免校准方法，更在跨任务能力均衡性上击败了在通用 C4 语料库上校准的强基线，且**对全部专家打分仅需 `0.22–2.06 秒`**。
* **`EvoESAP` 在高稀疏度开放式生成上取得显著增益**：在 `7B–30B` SMoE 模型、`25%` 与 `50%` 专家稀疏度下，`EvoESAP` 搜索出的非均匀层间分配一致优于均匀剪枝（Uniform Pruning），特别是在 `50%` 稀疏度下将开放式数学推理基准 **`MATH-500` 准确率提升高达 `+19.6%`**，同时保持多选任务竞争力。

#### 💡 与我们研究方向的闭环关联 (Connection to Our Research)
* **与我们的 `Capacity-Aware-MoE`、`Unified-MoE-Compression`、`awesome-mixture-of-experts`、`efficient_ads` 及 `ModelLesion` 形成直接闭环**：
  1. `EvoESAP` 原文明确将 `REAP`（我们此前重点追踪并对比的路由加权专家剪枝准则）等层内准则作为即插即用底座。我们可以直接把 `AIMER` 的免校准 $\ell _ 1 / \ell _ 2$ 权重集中度先验与 `EvoESAP` 的 `ESAP` 投机接受率代理集成进 `Capacity-Aware-MoE` 与 `Unified-MoE-Compression`；
  2. 在 `ModelLesion` 与 `LLM-Drop` 的跨层非均匀深度/宽度预算分配中，`ESAP` 提供了一个比普通交叉熵损失（PPL）对长程自回归生成退化敏感得多的有界代理指标。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Evolutionary Non-Uniform Layerwise Token/Expert Capacity Allocation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-10-01_ai_paper_notes.md`


---

### 3.2 [2026-10-01] FocusVLA & Navigation Heads: Modality Cascaded Focus Attention & Zero-Overhead Attention-Head Path Deviation Detection in VLAs (`arXiv:2603.28740` & `arXiv:2603.13782`)
* **论文标题**：
  1. *FocusVLA: Focused Visual Utilization for Vision-Language-Action Models* (`arXiv:2603.28740`)
  2. *Your Vision-Language-Action Model Already Has Attention Heads For Path Deviation Detection* (`arXiv:2603.13782`)
* **核心关键词**：`vla`, `vision-language-action`, `focusvla`, `navigation heads`, `path deviation`, `hallucination detection`, `robotic manipulation`, `rollback`

#### 📌 核心痛点与研究动机 (Motivation & Pain Points)
当前自回归与流匹配视觉-语言-动作（VLA）模型在真实机器人部署中面临“视觉利用不足”与“轨迹偏离幻觉”双重挑战：
1. **架构偏置导致 VLA 走“语言/历史动作捷径”而忽视细粒度视觉（`FocusVLA` 动机）**：`FocusVLA` 通过实证剖析指出，现有 VLA 性能受限的主要原因并非视觉编码器表征质量差，而是**视觉信息如何被策略网络利用**——（a）因果自注意力偏置使模型倾向于依赖语言先验捷径而忽略视觉细节；（b）过多的视觉 Token 稀释了注意力焦点；（c）任务无关的背景视觉噪声严重干扰动作生成。
2. **检测 VLA 轨迹偏离幻觉通常需要昂贵的外部 Critic（`Navigation Heads` 动机）**：当机器人在长程导航或操作中因视觉推理幻觉发生路径偏离（Path Deviation）时，传统方法必须额外训练外部价值网络（Critic）或运行多次前向不确定性采样，在端侧机器人上引入显著的计算与延迟负担。

#### ⚙️ 核心机制与数学公式推导 (Core Mechanism & Mathematical Formulation)
**第一部分：`FocusVLA` 的模态级联注意力（Modality Cascaded Attention）与动态聚焦调制（Focus Attention）**  
为彻底切断动作 Token 绕过视觉特征直接复制语言/本体感觉捷径的通路，`FocusVLA` 构建了**模态级联注意力（Modality Cascaded Attention）**：强制信息流按 $\text{Language Instruction } L \longrightarrow \text{Visual Patches } V \longrightarrow \text{Action Queries } A$ 的级联拓扑传递。在此基础上，**Focus Attention** 引入可学习的任务相关度门控 $g _ i = \sigma\left(f _ \psi(v _ i, \bar{q} _ L)\right) \in [0, 1]$ ，动态筛选 Top- $K$ 视觉块并对其键/值表征施加显式调制以抑制背景噪声：

$$
\widetilde{\mathrm{Attn}}\left(Q _ A, K _ V, V _ V\right) = \mathrm{Softmax}\left(\frac{Q _ A K _ {V, S}^\top}{\sqrt{d _ k}} + \log\left(g _ S + \epsilon\right)\right) \left(g _ S \odot V _ {V, S}\right), \quad S = \mathrm{TopK}\left(\lbrace g _ i \rbrace _ {i=1}^N, K\right)
$$

**第二部分：`Navigation Heads` 的内生时空因果监控与零开销安全回滚（Rollback）**  
`arXiv:2603.13782` 发现：在冻结 VLA 上千个注意力头中，存在极少数天然捕捉“历史视觉序列 $V _ {t-H:t}$ 与语言指令 $L$ 之间时空因果对齐”的注意力头，称为**导航头（Navigation Heads）** $\mathcal{H} _ {\text{nav}} = \lbrace (\ell _ 1, h _ 1), (\ell _ 2, h _ 2), (\ell _ 3, h _ 3) \rbrace$ （仅需 $|\mathcal{H} _ {\text{nav}}| = 3$ 个头）。在每次常规推理前向传播中，直接读取这 3 个头的跨模态注意力熵或时空对角线聚焦度 $z _ t^{(k)}$ ，构造零额外前向开销的实时异常检测统计量 $\mathcal{D} _ {\text{nav}}(t)$ ：

$$
\mathcal{D} _ {\text{nav}}(t) = \sum _ {k=1}^{3} w _ k \cdot \Phi\left(A _ {t}^{(\ell _ k, h _ k)}\left(\text{Action} \to V _ {t-H:t} \cup L\right)\right), \quad \pi _ {\text{exec}}(s _ t) = \begin{cases}
\pi _ {\text{VLA}}(s _ t), & \text{if } \mathcal{D} _ {\text{nav}}(t) \le \tau _ {\text{dev}} \cr
\pi _ {\text{RL-Rollback}}(s _ t), & \text{if } \mathcal{D} _ {\text{nav}}(t) > \tau _ {\text{dev}}
\end{cases}
$$

一旦 $\mathcal{D} _ {\text{nav}}(t) > \tau _ {\text{dev}}$ 判定发生路径偏离幻觉，系统立即旁路（Bypass）重型 VLA，切换至超轻量强化学习策略 $\pi _ {\text{RL-Rollback}}$ 执行最短路径安全回滚至正常轨迹流形。

#### 🎨 算法架构图与实现伪代码 (Architecture & Pseudocode)
```
====================================================================================================
   FocusVLA (模态级联聚焦注意力) + Navigation Heads (3 头零开销异常检测与 RL 回滚) (arXiv:2603.28740 & 13782)
====================================================================================================

  [Language L] ──► [Modality Cascaded Attention] ──► [Focus Attention: Select & Modulate Top-K Visual Patches]
                                                                     │
                                                                     ▼
                                                   [Frozen/Active VLA Backbone Forward]
                                                                     │
                       ┌─────────────────────────────────────────────┴──────────────────────────────────────┐
                       ▼ (直接复用前向传播中间张量，零额外计算开销)                                         ▼
        [Monitor 3 Intrinsic "Navigation Heads" H_nav]                                        [Normal VLA Action a_t]
        • 实时计算偏离分数 D_nav(t) (44.6% 检出率 @ 11.7% FPR)
                       │
          (若 D_nav(t) > τ_dev 触发异常告警)
                       ▼
        [Bypass Heavy VLA ──► Trigger Lightweight RL Policy π_RL-Rollback 执行最短路径安全回滚!]
====================================================================================================
```

#### 📊 实验指标与核心结论 (Experimental Results & Key Takeaways)
* **`FocusVLA` 提升精细操作与收敛速度**：在仿真与真实世界机器人基准上，`FocusVLA` 通过切断非视觉捷径并显式抑制无关背景噪声，在灵巧操作任务上大幅提升任务成功率并显著加快训练收敛速度。
* **仅监控 `3 个注意力头` 即实现高效实时异常检测与物理机器人回滚**：在超过一千个注意力头中，**仅组合 3 个 `Navigation Heads` 即可在不增加任何额外计算开销的前提下，实现 `44.6%` 的路径偏离检测率与 `11.7%` 的低误报率（False-Positive Rate）**，并在真实物理机器人上验证了“检测-旁路-轻量 RL 最短路径回滚”全链路的鲁棒性。

#### 💡 与我们研究方向的闭环关联 (Connection to Our Research)
* **直接赋能 `Axon V2` (`Pillar 1` & `Pillar 4`)、`VLADrop` 与 `TraceCraft`**：
  1. `Navigation Heads` 揭示的“无需外部 Critic、直接利用模型内部 3 个特定注意力头信号触发安全回滚”机制，可直接作为 `Axon V2` 循环 ODE 求解器（`axon/layers/looped_ode.py`）与动态提前退出（Early-Exit）的**零开销在线置信度看门狗（Watchdog）**；
  2. `FocusVLA` 的模态级联注意力为我们解决 VLA 在高压缩率下退化为“盲目开环动作记忆”提供了结构级约束。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Action-Conditioned Query-Guided Visual Token Routing Gate)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-10-01_ai_paper_notes.md`


---

### 3.3 [2026-09-30] ACPruner & SCOPD: Visual Token Pruning as Biased Attention Coverage Maximization & Sparse-Context On-Policy Self-Distillation (`arXiv:2609.34558` & `arXiv:2609.34044`)
* **论文标题**：
  1. *ACPruner: Visual Token Pruning as Biased Attention Coverage Maximization in LVLMs* (`arXiv:2609.34558`)
  2. *SCOPD: Sparse-Context On-Policy Self-Distillation for Efficient Vision-Language Models* (`arXiv:2609.34044`)
* **核心关键词**：`token pruning`, `visual token pruning`, `acpruner`, `scopd`, `coverage maximization`, `on-policy self-distillation`, `vlm`, `multimodal`

#### 📌 核心痛点与研究动机 (Motivation & Pain Points)
昨天我们精读的 `CoverPruner` (`arXiv:2609.03158`) 证明了纯视觉空间的 $k$ -Medoids 覆盖优化优于盲目 Top- $k$ 显著性截断，但在复杂视觉问答（如细粒度 OCR、图表定位、空间指代推理）中仍面临两大根本瓶颈：
1. **无偏几何覆盖与跨模态任务意图的错位（Unbiased Coverage vs. Query Intent）**：如果所有图像区域按等权重做空间覆盖，大量背景纹理 Token 仍会挤占有限预算 $K$ ，导致与用户文本指令强相关的微小前景区域采样不足；
2. **剪枝后的“表征-利用鸿沟（Representation-Utilization Gap）”**：即使剪枝算法把关键视觉 Token 保留了下来，预训练于稠密完整网格（Full Dense Grid）的 LLM 解码器在面对突然缺失 85%–90% 节点的稀疏上下文（Sparse Context）时，其深层自注意力路由会发生分布偏移，无法有效提取稀疏存活 Token 中的信息。

#### ⚙️ 核心机制与数学公式推导 (Core Mechanism & Mathematical Formulation)
**第一步：`ACPruner` 的偏置注意力覆盖最大化（Biased Attention Coverage Maximization）**  
给定 $N$ 个视觉 Token 表征 $X = \lbrace x _ 1, \dots, x _ N \rbrace$ 及跨模态文本指令对第 $i$ 个视觉 Token 的先验注意力重要性权重 $\pi _ i \in (0, 1)$ （归一化后 $\sum _ {i=1}^N \pi _ i = 1$ ）。定义对称正定的特征-空间复合亲和核 $K _ {ij} = \exp\left(-\frac{1 - \cos(x _ i, x _ j)}{\tau _ f} - \frac{\lVert p _ i - p _ j \rVert _ 2^2}{\tau _ s}\right)$ 。`ACPruner` 将子集选择 $S \subseteq \lbrace 1, \dots, N \rbrace$ （ $|S| = K$ ）构建为最大化**先验注意力加权的饱和覆盖效用函数** $\mathcal{F} _ {\text{AC}}(S)$ ：

$$
\max _ {S \subseteq \mathcal{V}, |S| = K} \mathcal{F} _ {\text{AC}}(S) = \sum _ {i=1}^N \pi _ i \cdot \phi\left(\max _ {j \in S} K _ {ij} + \beta \sum _ {j \in S} K _ {ij}\right)
$$

其中 $\phi(u) = \log(1 + u)$ 为严格凹单调饱和函数， $\beta > 0$ 平衡极值代表性（Facility Location）与局部密度覆盖。由于 $\mathcal{F} _ {\text{AC}}(S)$ 满足单调非负次模性（Monotone Submodularity），在每一步贪心选择中选取使边际覆盖增益 $\Delta _ {\text{AC}}(e \mid S _ t) = \mathcal{F} _ {\text{AC}}(S _ t \cup \lbrace e \rbrace) - \mathcal{F} _ {\text{AC}}(S _ t)$ 最大的 Token，即可保证 $\left(1 - 1/e\right)$ 最优近似界。

**第二步：`SCOPD` 的稀疏上下文在线自蒸馏（Sparse-Context On-Policy Self-Distillation）**  
为弥合“表征-利用鸿沟”，`SCOPD` 不使用任何外部人工标注答案，而是让共享参数 $\theta$ 的模型在**剪枝后的稀疏视觉上下文** $\tilde{V} = \mathrm{Prune}(V; K)$ 下自回归采样生成在线推理序列 $y \sim p _ \theta(\cdot \mid \tilde{V}, Q)$ （On-Policy Rollout）。随后，将同一条自生成序列 $y$ 喂给**拥有完整视觉上下文 $V$ 的冻结教师分支** $p _ {\text{tea}}(\cdot \mid V, Q)$ ，最小化在线轨迹上的逐位置反向 KL 散度与隐状态余弦对齐损失：

$$
\mathcal{L} _ {\text{SCOPD}}(\theta) = \mathbb{E} _ {y \sim p _ \theta(\cdot \mid \tilde{V}, Q)} \left[ \sum _ {t=1}^{|y|} \mathrm{KL}\left( p _ {\text{tea}}(\cdot \mid y _ {<t}, V, Q) \middle\Vert p _ \theta(\cdot \mid y _ {<t}, \tilde{V}, Q) \right) + \lambda _ {\text{hid}} \left(1 - \cos\left(h _ t^{\text{tea}}, h _ t^{\text{stu}}\right)\right) \right]
$$

#### 🎨 算法架构图与实现伪代码 (Architecture & Pseudocode)
```
====================================================================================================
     ACPruner (偏置注意力覆盖选点) + SCOPD (稀疏上下文在线自蒸馏) 协同流水线 (arXiv:2609.34558 & 34044)
====================================================================================================

  [Full Visual Tokens V (N=576)] + [Text Query Q]
                 │
                 ├──► (Branch A: Full-Context Teacher, Frozen) ──────────────────────┐
                 │    完整保留 576 个视觉 Token，仅对学生生成的在线轨迹 y 计算参考分布   │
                 │                                                                   ▼
                 └──► (Branch B: ACPruner Biased Coverage Selection)        [On-Policy KL + Hidden
                      • 计算指令先验权重 π_i 与复合核 K_ij                   Alignment Loss L_SCOPD]
                      • 次模贪心选出 K=64 个兼顾指令焦点与全局覆盖的锚点 ~V          ▲
                                          │                                          │
                                          ▼                                          │
                      [Student On-Policy Rollout: y ~ p_θ(· | ~V, Q)] ───────────────┘
                      在稀疏视觉上下文 ~V 上自回归采样生成推理链，彻底消除训练-推理由稠密转稀疏的分布失配
====================================================================================================
```

#### 📊 实验指标与核心结论 (Experimental Results & Key Takeaways)
* **极低保留率下的性能飞跃**：在 `LLaVA-NeXT-7B` 与 `Qwen2.5-VL-7B` 上，当视觉 Token 剪掉 **88.9%**（仅保留 `64/576` 个 Token）时，单独使用 `ACPruner` 即可在 10 个多模态基准上保留 **97.4%** 的原始精度（超越 `FastV`、`SparseVLM` 与无偏 `CoverPruner` 达 **+1.8–4.6 pp**）。

#### 💡 与我们研究方向的闭环关联 (Connection to Our Research)
* **赋能 `SparseUnifiedModel`、`VLADrop` 与 `Axon V2` (`Pillar 1: RL-HiSTrim`)**：我们在 `VLADrop` 和 `Axon V2` 中对多视角相机图像做 Token 剪枝时，常观察到当保留率压至 ≤ 20% 时动作专家在精细抓取阶段会出现几厘米的定位偏差（正是 `SCOPD` 揭示的“表征-利用鸿沟”）。将 `ACPruner` 的跨模态偏置次模核与 `SCOPD` 的在线稀疏上下文自蒸馏引入 `axon/models/vla_pruner.py` 与 `sparse_umm/token_pruning.py`，可在不改动推理架构的前提下彻底抚平高倍率视觉剪枝带来的特征断层。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Biased Attention Coverage Maximization for Token Routing)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-30_ai_paper_notes.md`


---

### 3.4 [2026-09-30] Dynamic Flow, Static Graph & DORA: KV Cache Reuse on Static NPU Graphs & Dynamic Online RL Token Pruning (`arXiv:2609.34727` & `arXiv:2609.34325`)
* **论文标题**：
  1. *Dynamic Flow, Static Graph: KV Cache Reuse for Efficient LLM Serving on Mobile NPUs* (`arXiv:2609.34727`)
  2. *DORA: Dynamic Online Reinforcement Agent for Token Pruning in Vision Transformers* (`arXiv:2609.34325`)
* **核心关键词**：`kv cache`, `static graph`, `mobile npu`, `dora`, `token pruning`, `reinforcement learning`, `histrim`, `dynamic routing`

#### 📌 核心痛点与研究动机 (Motivation & Pain Points)
在端侧移动 NPU、车载芯片或开启 `torch.compile(..., mode="reduce-overhead")` / CUDA Graphs 的云端推理引擎中，**“算法层的动态稀疏性（Dynamic Sparsity）”与“硬件编译层的静态图约束（Static Execution Graph）”存在尖锐矛盾**：
1. **动态 KV 复用与局部重计算破坏静态张量形状（`arXiv:2609.34727`）**：在多轮对话或 RAG 检索中，不同文档块（Non-Prefix Chunks）的复用长度和需重计算交叉注意力的 Token 数量 $M _ {\text{recomp}}$ 随请求动态变化。若每次按实际 $M _ {\text{recomp}}$ 编译新图，NPU 图重编译耗时高达数秒；若退回全量 Prefill，又浪费 80% 算力；
2. **固定剪枝率无法适应样本级难度波动（`DORA` `arXiv:2609.34325`）**：现有视觉 Token 剪枝对简单纯色背景图与复杂密集遮挡图强行施加相同的固定保留率 $\rho$ ，导致简单样本算力浪费、复杂样本精度崩塌。

#### ⚙️ 核心机制与数学公式推导 (Core Mechanism & Mathematical Formulation)
**第一部分：`Dynamic Flow, Static Graph` 的静态图分桶掩码嵌入与计算-存储协同调度**  
给定预编译的固定大小静态计算图粒度 $B _ {\text{tile}}$ （如 `64` 或 `128` Token）。对于总长为 $L$ 的上下文，其中非连续可复用块集合为 $\mathcal{C} _ {\text{reuse}}$ ，需重计算的关键交叉注意力 Token 集合为 $\mathcal{I} _ {\text{recomp}}$ （通过缓存的浅层键值重要度快速选出）。算法将 $|\mathcal{I} _ {\text{recomp}}|$ 向上对齐至固定倍数 $m \cdot B _ {\text{tile}}$ ，并通过硬件级 Gather-Mask 算子构建静态形状张量 $X _ {\text{static}} \in \mathbb{R}^{B _ {\text{tile}} \times d}$ ，同时将 FlashAttention 输出更新严格限制在动态索引集上：

$$
K _ {\ell}[\mathcal{I} _ {\text{recomp}}] \leftarrow X _ {\text{static}} W _ K^{(\ell)}, \quad V _ {\ell}[\mathcal{I} _ {\text{recomp}}] \leftarrow X _ {\text{static}} W _ V^{(\ell)}, \quad O _ {\text{static}} = \mathrm{Softmax}\left(\frac{Q _ {\text{static}} K _ {\ell}^\top}{\sqrt{d _ k}} + M _ {\text{causal-pad}}\right) V _ {\ell}
$$

其中 $M _ {\text{causal-pad}} \in \lbrace 0, -\infty \rbrace^{B _ {\text{tile}} \times L _ {\text{max}}}$ 屏蔽尾部对齐 Padding 位，从而在 **100% 零图重编译（Zero Graph Recompilation）** 的前提下实现任意位置的动态 KV 缓存拼接与选择性重计算。

**第二部分：`DORA` 的分层在线强化学习动态剪枝智能体**  
`DORA` 将冻结骨干网第 $\ell$ 层的 Token 剪枝建模为分层马尔可夫决策过程（Hierarchical MDP）：高层策略 $\pi _ {\phi}^{\text{high}}(r _ \ell \mid s _ \ell)$ 根据当前层注意力熵与特征方差状态 $s _ \ell$ 动态决定该层的**样本专属保留率 $r _ \ell \in \lbrace 0.3, 0.5, 0.7, 0.9, 1.0 \rbrace$ **；低层策略 $\pi _ {\psi}^{\text{low}}(m _ \ell \mid X _ \ell, r _ \ell)$ 输出个体 Token 的保留评分，最大化兼顾任务精度与延迟惩罚的在线奖励函数：

$$
\mathcal{R}(X, \lbrace r _ \ell \rbrace _ {\ell=1}^L) = -\mathcal{L} _ {\text{task}}\left(\hat{y}(\lbrace r _ \ell \rbrace), y\right) - \lambda _ {\text{flops}} \cdot \max\left(0, \frac{1}{L}\sum _ {\ell=1}^L \prod _ {j=1}^\ell r _ j - \rho _ {\text{target}}\right)^2
$$

#### 🎨 算法架构图与实现伪代码 (Architecture & Pseudocode)
```
====================================================================================================
   Dynamic Flow, Static Graph (静态图动态 KV 复用) + DORA (分层 RL 动态剪枝) (arXiv:2609.34727 & 34325)
====================================================================================================

  [Input Context / Visual Stream]
         │
         ├──► [DORA High-Level RL Actor π_high] ──► 根据样本复杂度动态决策当前层保留预算 r_l
         │
         └──► [Static-Graph Bucket Gather (Align to B_tile = 64)]
              将动态数量的存活/需重算 Token 打包进固定形状静态张量 X_static + 掩码 M_causal-pad
                                       │
                                       ▼
              [Zero-Recompilation NPU / CUDA Graph Execution] (100% 硬件静态图加速 + 动态算力节省)
====================================================================================================
```

#### 📊 实验指标与核心结论 (Experimental Results & Key Takeaways)
* **端侧静态图 NPU 首字延迟骤降**：在高通骁龙 8 Elite（Hexagon NPU）与端侧 SoC 上运行 `Qwen2.5-3B/7B` 与 `Llama-3.2-3B`，`Dynamic Flow, Static Graph` 完全消除了动态形状引发的 CPU 回退与图重编译，相比全量重算将多文档 RAG 首字延迟（TTFT）降低 **3.4×–4.8×**，能耗降低 **62%**。
* **`DORA` 自适应超越静态剪枝**：在 ImageNet 与下游多模态视觉任务上，在相同平均 FLOPs 预算（削减 50% 计算量）下，`DORA` 凭借样本级动态保留率分配比固定剪枝率基线（`ToMe`、`EViT`）提升 **+1.4%–2.3%** 准确率。

#### 💡 与我们研究方向的闭环关联 (Connection to Our Research)
* **直接对标并赋能 `Efficient Ads` (`RL-HiSTrim`)、`Axon V2` (`Zero-Sync CUDA Graph`) 与 `Router-Tuning`**：我们在 `Efficient Ads` 和 `Axon V2` 中正是使用强化学习策略（GRPO）学习逐层 Token 剪枝率，并在 `Axon V2` 中強調 `CUDA Graph` 零同步快路径（Zero-Sync Fast-Path）。`arXiv:2609.34727` 的固定分桶 Tile 对齐 + 掩码注入机制，为我们把 `DORA`/`RL-HiSTrim` 的样本级动态保留率直接编译进静态 `CUDA Graph` / `TPU XLA` 提供了完美的工程范式。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Dynamic Online RL Advantage-Guided Token Routing)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-30_ai_paper_notes.md`


---

### 3.5 [2026-09-29] ✂️ *CoverPruner & SFPruner: Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization & Single-Forward Ridge Leverage*
> 🏷️ **核心关键词**：Visual Token Pruning · Representational Coverage Maximization (RCM) · Ridge Leverage Score · High-Resolution MLLMs  
> 🔗 **arXiv 链接**：[`arXiv:2609.03158`](https://arxiv.org/abs/2609.03158) (`CoverPruner`) & [`arXiv:2607.23046`](https://arxiv.org/abs/2607.23046) (`SFPruner`)

```
  高分辨率视觉 Token 集合 V = {v_1..v_N}
       ├──► [ CoverPruner (2609.03158) ] : 语义显著性 q_i × 设施选址最大覆盖 max_{j ∈ S} sim(v_i, v_j) ──► 确保每个被剪 Token 有高相似“代言人”
       └──► [ SFPruner    (2607.23046) ] : 语义引导核矩阵岭杠杆分数 τ_i(λ) + 单次方向性排序掩码    ──► 零迭代 O(N d^2) 单次前向去冗余保留子集 S*
```

#### 🎯 背景与痛点 (Problem Statement)
现有视觉语言大模型（VLM / MLLM）的视觉 Token 剪枝算法大多遵循“独立显著性排序（Top- $k$ Salience Ranking）”范式——即根据跨模态注意力得分或范数选出前 $K$ 个最高分的视觉 Token。然而，高分辨率图像中的高显著性 Token 往往高度聚集在少数显著前景物体内部（形成严重的**空间与语义聚团同质化**），导致被保留的 $K$ 个 Token 彼此高度冗余，而大面积次显著但包含关键空间关系、文字细节或环境上下文的视觉区域却“无人代言（No One Speaks for the Pruned）”而被整体抹除；与此同时，传统的子模函数或 $k$ -Medoids 多样性子集选择需要 $O(K N^2)$ 串行贪心迭代，在 GPU 上引入不可接受的推理延迟。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **表征覆盖最大化目标（Representational Coverage Maximization, `CoverPruner`）**：
  设视觉 Token 表征集合为 $\mathcal{V} = \lbrace v _ 1, \dots, v _ N \rbrace \subset \mathbb{R}^d$ ，跨模态指令相关性先验权重为 $w _ i \ge 0$ 。`CoverPruner` 不再最大化保留子集 $\mathcal{S} \subset \mathcal{V}$ （ $\lvert \mathcal{S} \rvert = K$ ）的孤立得分之和，而是最大化**全体原始视觉 Token 在保留子集 $\mathcal{S}$ 上的加权最近邻表征覆盖度（Weighted Facility Location Coverage）**：

$$
\mathcal{S}^{\star} = \arg\max _ {\mathcal{S} \subset \mathcal{V}, \lvert \mathcal{S} \rvert = K} \sum _ {i=1}^{N} w _ i \cdot \max _ {j \in \mathcal{S}} \left( \frac{\langle v _ i, v _ j \rangle}{\lVert v _ i \rVert _ 2 \lVert v _ j \rVert _ 2 + \epsilon} \right)
$$

* **单次前向语义引导岭杠杆打分（Single-Forward Ridge Leverage Score, `SFPruner`）**：
  为彻底消除组合优化的串行迭代开销，`SFPruner` 将语义显著性对角矩阵 $W = \mathrm{diag}(w _ 1, \dots, w _ N)^{1/2}$ 融入特征矩阵 $\tilde{V} = W V \in \mathbb{R}^{N \times d}$ ，通过单次前向闭式计算每个视觉 Token 在主成分子空间中的**统计岭杠杆分数（Ridge Leverage Score）** $\tau _ i(\lambda)$ ，并结合排序方向掩码（Ranking-Based Directional Masking）一步抑制高共线性冗余邻居：

$$
\tau _ i(\lambda) = \tilde{v} _ i^{\top} \left( \tilde{V}^{\top} \tilde{V} + \lambda I _ d \right)^{-1} \tilde{v} _ i, \quad \tilde{\mathcal{S}} _ {\text{SF}}(i) = \tau _ i(\lambda) \cdot \left( 1 - \max _ {j : w _ j > w _ i} \cos(\tilde{v} _ i, \tilde{v} _ j) \right)
$$

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 LLaVA-NeXT、Qwen2.5-VL 与 InternVL-2.5 等高分辨率多模态模型上，当剪除 **80%–88.9% 视觉 Token**（仅保留 64–128 个 Token）时，`CoverPruner` 与 `SFPruner` 在细粒度 OCR（DocVQA、TextVQA）与空间关系推理（MMBench、GQA）上比 FastV、SparseVLM 与 PruMerge 平均提升 **+3.4%–+5.2%**，且算子本身在 GPU 上仅耗时 **< 0.8 ms**（零迭代并行矩阵求逆）。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：直接印证并补充了我们在 ***Understanding and Harnessing Sparsity for Unified Multimodal Models***（`TMLR 2026`, `SparseUnifiedModel`）、***ModelLesion (`EXP-E14` Token 信息密度 + 中心化分岔召回)*** 以及 ***Efficient Ads (`HisTrim`)*** 中的多模态与长序列去冗余思想。
* **落地到 `VLADrop`、`axon_v2`、`ModelLesion` 与 `efficient_ads`**：在 `VLADrop` (`models/pi0.5/`, `models/openvla-oft/`) 与 `ModelLesion` (`token_info_density_compressor.py`) 中，可直接引入 $d \times d$ 协方差矩阵求逆的**单次前向岭杠杆分数 $\tau _ i(\lambda)$ **，用闭式二阶子空间独立性度量替代启发式余弦去重。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Coverage-Preserving Token Routing & Barycentric Aggregation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 3.6 [2026-09-29] 🦾 *DEE-VLA: Decoupled Early Exits for Task-Dependent Compute Allocation in Flow-Matching VLAs*
> 🏷️ **核心关键词**：Vision-Language-Action (VLA) · Flow Matching · Decoupled Early Exits · Dynamic Compute Allocation  
> 🔗 **arXiv 链接**：[`arXiv:2609.29382`](https://arxiv.org/abs/2609.29382)

```
  多模态观测 (I_t, l) ──► [ VLM 感知主干 (层 1..L_vlm) ] ──► 动态早退层 l_vlm* (自由空间移动早退，精细对准深层)
                                                                  │ (跨模块KV桥接投影)
                                                                  ▼
                     [ Flow-Matching Action Expert (层 1..L_act, 积分步 k=1..K) ]
                                                                  ├──► 动作网络深度早退 l_act*(k)
                                                                  └──► 速度场曲率收敛早退步 K* ──► 实时机器人控制动作 a_t
```

#### 🎯 背景与痛点 (Problem Statement)
现有的视觉-语言-动作（VLA）基础模型（如 $\pi _ 0$ 、 $\pi _ {0.5}$ 、GR00T）通常由百亿级参数的多模态 VLM 主干与数亿参数的流匹配（Flow-Matching）动作专家（Action Expert）组成。以往的早退（Early Exit）或层剪枝方法往往将 VLM 主干深度与动作专家深度**强行绑定（Coupled Depth Scaling）**，或者对一整段轨迹的所有时间步施加相同的静态深度预算。然而，机器人操纵任务具有显著的**时空异质性（Spatio-Temporal Heterogeneity）**：在粗粒度场景理解已完成的抓取接近阶段，VLM 主干仅需浅层表征即可维持语义定位，而进入毫米级插孔（Peg-in-Hole）接触瞬间，VLM 无需重算深层语义但 Action Expert 却需要更深的网络层数与更多的流匹配 ODE 修正步来解析高频接触动力学。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **三轴解耦动态计算空间（Tri-Axis Decoupled Compute Space）**：
  `DEE-VLA` 将单次控制周期的推理算力分解为三个可独立调节的正交自由度：（1）VLM 主干退出层 $l _ {\text{vlm}} \in \lbrace 1, \dots, L _ {\text{vlm}} \rbrace$ ；（2）第 $k$ 个流匹配步中 Action Expert 的退出层 $l _ {\text{act}}^{(k)} \in \lbrace 1, \dots, L _ {\text{act}} \rbrace$ ；（3）流匹配歐拉积分的总终止步数 $K^{\star} \in \lbrace 1, \dots, K _ {\max} \rbrace$ 。
* **跨层隐状态余弦稳定性与速度场曲率早退门控**：
  在 VLM 主干内部，当相邻两层视觉-语言融合隐状态的余弦相似度超过语义收敛阈值 $\tau _ {\text{vlm}}$ 时提前退出并通过轻量级层对齐投影器生成 KV 缓存；在流匹配 Action Expert 内部，同时监测跨层速度预测残差与跨 ODE 步的**流场直线性曲率（Flow Straightness Curvature）**：

$$
\mathcal{E} _ {\text{depth}}^{(k)}(l) = \frac{\lVert v _ {\theta}^{(l)}(x _ {t _ k}, t _ k) - v _ {\theta}^{(l-1)}(x _ {t _ k}, t _ k) \rVert _ 2}{\lVert v _ {\theta}^{(l-1)}(x _ {t _ k}, t _ k) \rVert _ 2 + \epsilon} \le \delta _ {\text{act}}, \quad \mathcal{E} _ {\text{flow}}(k) = \lVert v _ {\theta}^{\star}(x _ {t _ k}, t _ k) - v _ {\theta}^{\star}(x _ {t _ {k-1}}, t _ {k-1}) \rVert _ 2 \le \delta _ {\text{ode}}
$$

  一旦 $\mathcal{E} _ {\text{flow}}(k) \le \delta _ {\text{ode}}$ （表明当前局部流场已呈直线匀速轨迹），立即跳过剩余 ODE 积分步并利用一阶欧拉外推直接输出终端动作块 $\hat{x} _ 1 = x _ {t _ k} + (1 - t _ k) v _ {\theta}^{\star}(x _ {t _ k}, t _ k)$ 。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 LIBERO（Spatial / Object / Goal / Long）与真机双臂灵巧操作任务上，`DEE-VLA` 在成功率与全深度 10-NFE 基线持平（甚至因减少自由空间过拟合而提升 **+0.8%**）的同时，平均削减了 **54.2% 的总 FLOPs** 与 **49.6% 的端到端控制延迟**，自动展现出“自由空间巡航浅层少步、接触操作阶段深层精细积分”的涌现算力分配规律。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：与我们在 `axon_v2` / `VLADrop` 中构建的 **Tri-Orthogonal Depth-Width-Step Compression (`G19`)**、**Once-for-All Switchable Multi-Gear Loops** 以及 **Looped VLA 动态停止准则 $K(s _ t, t, m)$ ** 完全同源！
* **落地到 `axon_v2` 与 `VLADrop` (`VLM-Compression`)**：可在 `axon/layers/looped_ode.py` 与 `VLADrop/models/pi0.5/` 中直接融合 `DEE-VLA` 的双判据 $\left( \mathcal{E} _ {\text{depth}}^{(k)}(l), \mathcal{E} _ {\text{flow}}(k) \right)$ ，将 VLM 编码器深度门控与 Action Expert 的 ODE 步曲率门控彻底解耦。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Decoupled Early-Exit Halting Gates across Multimodal & Action Towers)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 3.7 [2026-09-28] ✂️ *CLSE: Spectral Evolution-Guided Token Pruning in Multimodal Large Language Models*
> 🏷️ **核心关键词**：Multimodal Token Pruning · Cross-Layer Spectral Evolution · Discrete Cosine Transform (DCT) · Training-Free Compression  
> 🔗 **arXiv 链接**：[`arXiv:2606.24165`](https://arxiv.org/abs/2606.24165) (ECCV 2026)

```
  层 l-1 视觉隐状态 H^(l-1) ──► [ 通道维 DCT 频域投影 Φ ] ──► 频谱能量分布 P^(l-1)(ω) ┐
                                                                                      ├──► [ 跨层谱演化散度 D_CLSE(i) ] ──► Top-K 语义活跃视觉 Token 保留
  层 l   视觉隐状态 H^(l)   ──► [ 通道维 DCT 频域投影 Φ ] ──► 频谱能量分布 P^(l)(ω)   ┘
```

#### 🎯 背景与痛点 (Problem Statement)
现有多模态大模型（MLLM / VLM）免训练视觉 Token 剪枝方法（如 FastV、SparseVLM）大多依赖**单层静态注意力分数**（如第 $l$ 层文本对视觉 Token 的注意力权重 $A _ {t, v}^{(l)}$ ）。然而，由于 RoPE 旋转位置编码的远程衰减与视觉 Sink Token 现象，单层注意力极易受到空间位置偏置（Position Bias）误导——许多在当前层注意力得分较高但跨层表征几乎停止更新的“静态背景/锚点冗余 Token”被错误保留，而真正正在经历高频语义整合的关键局部视觉 Token 却被过早裁剪。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **跨层频域映射与归一化能量谱**：
  设第 $l$ 层第 $i$ 个视觉 Token 的隐状态向量为 $h _ i^{(l)} \in \mathbb{R}^d$ 。CLSE 首先通过正交离散余弦变换（DCT）基矩阵 $\Phi \in \mathbb{R}^{d \times d}$ 将特征变化量 $\Delta h _ i^{(l)} = h _ i^{(l)} - h _ i^{(l-1)}$ 投影至频域，计算频谱系数 $c _ i^{(l)} = \Phi h _ i^{(l)}$ ，并构造归一化频域能量分布：

$$
p _ {i, k}^{(l)} = \frac{\left( c _ {i, k}^{(l)} \right)^2}{\sum _ {m=1}^{d} \left( c _ {i, m}^{(l)} \right)^2 + \epsilon}, \quad k \in \lbrace 1, \dots, d \rbrace
$$

* **跨层谱演化散度（Cross-Layer Spectral Evolution Score）**：
  原文发现：真正参与跨模态语义推理的视觉 Token 在穿越浅层到中层 Transformer 时，其能量会从高频局部纹理分量向低频全局语义分量发生剧烈的**谱重分布（Spectral Redistribution）**；而背景冗余 Token 的频谱分布则保持停滞。因此定义第 $i$ 个视觉 Token 在第 $l$ 层的跨层谱演化显著性打分为对称 Jensen-Shannon 谱演化散度与残差平行演化幅度的乘积：

$$
\mathcal{S} _ {\text{CLSE}}^{(l)}(i) = \mathrm{JSD}\left( p _ i^{(l)} \parallel p _ i^{(l-1)} \right) \cdot \frac{\lVert h _ i^{(l)} - h _ i^{(l-1)} \rVert _ 2}{\lVert h _ i^{(l-1)} \rVert _ 2 + \epsilon}
$$

* **谱演化引导渐进剪枝**：
  在预设的剪枝过渡层 $l \in \mathcal{L} _ {\text{prune}}$ ，仅保留 $\mathcal{S} _ {\text{CLSE}}^{(l)}(i)$ 排名前 $K _ l$ 的语义活跃 Token，被裁剪的背景 Token 按频谱相似度加权合并至最近邻保留 Token 中以守恒低频能量。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 LLaVA-NeXT-7B/13B 与 Qwen2.5-VL-7B 上，免训练剪除 **75% 视觉 Token** 时，在 DocVQA、ChartQA、TextVQA 与 Video-MME 等 10 项多模态基准上平均保留了 **99.1%** 的原始全量精度，显著超越单层注意力剪枝基线（+3.8%），预填充（Prefill）FLOPs 降低 **68%**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：直接呼应我们在 ***Understanding and Harnessing Sparsity for Unified Multimodal Models***（`TMLR 2026`, `SparseUnifiedModel`）、***Demystifying When Pruning Works via Representation Hierarchies***（`ICML 2026`, `Pruning-on-Representations`）以及 ***Transformer-Geometry***（`EMNLP 2026`, `arXiv:2609.15975`）中提出的跨层表征几何演化理论。
* **落地到 `VLADrop` (`VLM-Compression`) 与 `efficient_ads` (`HisTrim`)**：在我们的 `VLADrop` 具身视觉编码器与 `HisTrim` 多阶段分层序列裁剪中，可将单层注意力打分升级为 **跨层平行/正交残差演化率 + 频域谱重分布散度 $\mathcal{S} _ {\text{CLSE}}^{(l)}$ **，用零额外参数的逐层残差差分替代易受位置偏置干扰的静态注意力权重。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Spectral Entropy Inflection Layer Allocation for Token Routing)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-28_ai_paper_notes.md`


---

### 3.8 [2026-09-28] ✂️ *ASL: Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference*
> 🏷️ **核心关键词**：Layer-Wise Token Pruning · Adaptive Layer Selection · Attention Variance · Long-Context LLM Inference  
> 🔗 **arXiv 链接**：[`arXiv:2601.07667`](https://arxiv.org/abs/2601.07667) (ACL 2026 Findings)

```
  输入长序列 X ──► 逐层前向传播 l=1..L ──► 实时监测注意力熵变与表征漂移率 η_l
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        ▼ (η_l 跌破相变阈值 τ: 语义路由已收敛)                     ▼ (η_l > τ: 仍在剧烈跨位置交互)
          [ 触发 ASL 单次 Token 剪枝 (One-Shot Selection) ]                [ 保持全长序列继续前向传播 ]
```

#### 🎯 背景与痛点 (Problem Statement)
现有的长上下文逐层 Token 剪枝方法（如 PyramidInfer、LazyLLM）通常采用**跨样本固定的剪枝层配置**（例如硬编码在第 4、8、16 层按固定比例裁剪 Token）。然而，不同复杂度与不同上下文长度的输入样本，其跨位置信息汇聚的完成深度截然不同：简单检索任务在第 6 层已完成关键信息聚焦，而多跳推理任务直到第 18 层仍在跨段落聚合线索。静态固定剪枝层要么在困难样本上过早剪断推理链，要么在简单样本上浪费大量冗余计算。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **跨层注意力方差与路由收敛度度量**：
  设第 $l$ 层查询窗口对上下文 Token 的平均注意力分布为 $\bar{\alpha}^{(l)} \in \Delta^{N-1}$ 。ASL 提出用注意力分布的**二阶方差锐度（Attention Variance Sharpness）**与相邻层注意力分布的 **余弦收敛度** 联合度量当前层是否已完成信息路由聚焦：

$$
\mathcal{C} _ l = \mathrm{Var}\left( \bar{\alpha}^{(l)} \right) \cdot \frac{\left\langle \bar{\alpha}^{(l)}, \bar{\alpha}^{(l-1)} \right\rangle}{\lVert \bar{\alpha}^{(l)} \rVert _ 2 \lVert \bar{\alpha}^{(l-1)} \rVert _ 2 + \epsilon}
$$

* **自适应剪枝层触发准则（Adaptive Layer Selection）**：
  当第 $l$ 层的聚焦收敛指数 $\mathcal{C} _ l$ 首次超过样本自适应阈值 $\tau _ {\text{ASL}}$ 且层间相对增幅趋于平缓（即 $\lvert \mathcal{C} _ l - \mathcal{C} _ {l-1} \rvert \le \delta$ ）时，ASL 判定该样本在层 $l^{\star}$ 已越过“信息收集—语义提纯相变点”，随即在层 $l^{\star}$ 触发 **One-Shot Token Selection**，一次性保留核心上下文子集 $\mathcal{I} _ {\text{keep}}$ ：

$$
l^{\star}(x) = \min \left\lbrace l \in \lbrace l _ {\min}, \dots, L \rbrace \middle| \mathcal{C} _ l(x) \ge \tau _ {\text{ASL}} \land \lvert \mathcal{C} _ l(x) - \mathcal{C} _ {l-1}(x) \rvert \le \delta \right\rbrace
$$

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 Llama-3.1-8B/70B 与 Qwen2.5-14B 上，针对 RULER、InfiniteBench 与 Needle-in-a-Haystack（128K 上下文）评测表明：在相同的 **2.4× 端到端推理加速比**下，ASL 比固定层级剪枝基线在多跳问答与长程聚合任务上平均提升 **+4.6 分**，彻底消除了静态早剪导致的“大海捞针丢失”现象。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：与我们在 ***Uncovering the Redundancy in Transformers via Layer Dropping***（`TMLR 2025`, `LLM-Drop`）、***Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers***（`EMNLP 2025`, `Router-Tuning-Mixture-of-Depths`）以及 ***Demystifying When Pruning Works via Representation Hierarchies***（`ICML 2026`, `Pruning-on-Representations`）中揭示的“语义表征相变层（Phase-Transition Layer）”高度吻合。
* **落地到 `LLM-Drop`、`ModelLesion` 与 `efficient_ads`**：可将 ASL 的样本级在线收敛准则 $\mathcal{C} _ l(x)$ 引入 `efficient_ads` 的 `HisTrim` 多阶段裁剪触发器以及 `LLM-Drop` 的动态跳过门控中，实现**按样本难度自适应推迟或提前剪枝触发层 $l^{\star}(x)$ **。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Marginal Information Gain Schedule for Dynamic-Depth MoD Layers)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-28_ai_paper_notes.md`


---

### 3.9 [2026-09-27] L2R: Low-Rank and Lipschitz-Controlled Routing for Mixture-of-Experts

* **论文信息**：Minghao Yang, Ren Togo, Guang Li, Takahiro Ogawa, Miki Haseyama (`arXiv:2601.21349`, 2026-01)
* **核心关键词**：MoE Routing Geometry、Low-Rank Latent Space、Lipschitz Continuity、Saturated Inner-Product Scoring (SIPS)、Multi-Anchor Routing

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|          L2R: Low-Rank & Lipschitz-Controlled MoE Routing Architecture            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|                        Token Hidden State h \in R^d                               |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 1. Shared Low-Rank Latent Projection (低秩路由子空间映射)|                |
|        |    z = P h \in R^r   (r << d, orthogonalized P P^T = I_r)|                |
|        |    Filters out high-dimensional isotropic noise         |                |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 2. Multi-Anchor Expert Prototypes (多锚点专家原型表示)   |                |
|        |    Each Expert e has M low-rank anchors: {u_{e,m}}_{m=1}^M               |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 3. Saturated Inner-Product Scoring (SIPS Lipschitz 控制) |                |
|        |    s_{e,m}(z) = \tau \cdot \tanh( <z, u_{e,m}> / (\tau \|z\|_\gamma) )   |
|        |    Explicitly bounds || \nabla_h s_e(h) ||_2 <= L_lip    |                |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|             SoftMax / Top-k Selection ---> Stable Expert Dispatch                 |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **高维线性路由的三大几何病态**：标准稀疏 MoE 普遍采用单层线性投影 $s(h) = W _ r h \in \mathbb{R}^N$ 作为路由器（Router）。作者从表示几何角度指出高维空间 $d \gg N$ 中的线性内积路由存在三大固有缺陷：
  1. **维度失配与噪声过拟合（Representation Mismatch）**：Token 隐状态 $h \in \mathbb{R}^d$ 包含了大量与任务路由无关的词法/位置高频噪声，全维内积导致路由决策极易受正交噪声方向干扰。
  2. **高维角度集中现象（Angular Concentration）**：随着层深增加，Transformer 隐状态落入狭窄的各向异性锥（Anisotropic Cone），不同专家路由向量与 $h$ 的余弦相似度高度趋同，导致门控分布扁平化或赢家通吃。
  3. **范数敏感与 Lipschitz 失控（Scale Sensitivity）**：当隐状态范数 $\Vert h\Vert _ 2$ 在深层或长序列中剧烈膨胀时，未受控的内积 $w _ e^\top h$ 会使 Softmax 进入指数饱和区，微小输入扰动即可引发离散 Top- $k$ 路由集合翻转（Routing Instability）。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **共享低秩潜空间路由投影（Low-Rank Latent Routing Space）**：
   引入行正交低秩投影矩阵 $P \in \mathbb{R}^{r \times d}$ （ $r \ll d$ ，例如 $d=2048, r=64$ ），将隐状态 $h$ 压缩至低秩判别子空间：

$$
z = P h \in \mathbb{R}^r, \qquad \mathcal{L} _ {\text{orth}} = \Vert P P^\top - I _ r \Vert _ F^2
$$

2. **饱和内积打分与显式 Lipschitz 边界控制（Saturated Inner-Product Scoring, SIPS）**：
   为消除隐状态径向范数 $\Vert h\Vert _ 2$ 暴涨导致的路由震荡，L2R 设计了带阻尼范数归一化与双曲正切饱和的打分算子：

$$
\phi _ {\text{SIPS}}(z, u _ e) = \tau \cdot \tanh\left( \frac{\langle z, u _ e \rangle}{\tau \left(\sqrt{\Vert z\Vert _ 2^2 + \epsilon^2}\right)^\gamma \left(\sqrt{\Vert u _ e\Vert _ 2^2 + \epsilon^2}\right)^\gamma} \right)
$$

   其中 $\tau > 0$ 控制饱和软边界， $\gamma \in [0, 1]$ 控制径向尺度不变性强度（当 $\gamma=1$ 时退化为受控余弦路由）。利用 $\text{sech}^2(x) \le 1$ 及正交投影 $\Vert P\Vert _ 2 = 1$ ，可严格证明打分函数对原始输入 $h$ 的梯度范数（即局部 Lipschitz 常数）存在显式解析上界：

$$
\left\lVert \nabla _ h \phi _ {\text{SIPS}}(P h, u _ e) \right\rVert _ 2 \le \Vert P\Vert _ 2 \cdot \frac{\Vert u _ e\Vert _ 2^{1-\gamma}}{\epsilon^\gamma} = L _ {\text{lip}}
$$

   从而从数学上保证了有界输入扰动 $\Vert\delta h\Vert _ 2 \le \delta$ 不会引发路由分数的剧烈跳变。
3. **多锚点专家表达（Multi-Anchor Routing）**：
   由于单个专家往往需要处理多模态或多子类语义簇，在低秩空间 $\mathbb{R}^r$ 中为每个专家分配 $M$ 个子锚点 $\lbrace u _ {e,m}\rbrace _ {m=1}^M \subset \mathbb{R}^r$ （参数量仅为 $N \times M \times r \ll N \times d$ ），通过 Log-Sum-Exp 软聚合计算专家总得分：

$$
s _ e(h) = \frac{1}{\beta} \log \sum _ {m=1}^M \exp\Big( \beta \cdot \phi _ {\text{SIPS}}(P h, u _ {e,m}) \Big)
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* **语言与视觉双模态全面验证**：在基于 **OLMoE** 的语言模型预训练/微调以及 **ImageNet** 视觉 MoE 骨干网络上，L2R 将路由器参数量削减 **60%–75%**，同时在相同激活专家预算下将下游任务困惑度（PPL）降低 `0.42–0.68`，ImageNet Top-1 准确率提升 `+1.3%`。
* **路由稳定性与负载均衡双升**：在对抗性高斯扰动测试下，L2R 的 Top- $k$ 路由翻转率（Routing Flip Rate）比标准线性 Router 降低 **47%**，专家负载熵（Routing Entropy）更加接近理想均匀分布，无需强依赖破坏主任务梯度的大权重 Load-Balancing 辅助损失。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
1. **与 *Router-Tuning* (EMNLP 2025) & *Capacity-Aware Inference* (ICLR 2026) 的直接耦合**：
   * 我们在 *Router-Tuning* 中提出仅微调轻量路由器即可解锁深层稀疏网络潜力，但在极低资源或长上下文微调中，全维线性路由器容易过拟合表面范数特征。将 L2R 的 **SIPS + 低秩多锚点路由** 作为 *Router-Tuning* 的参数化形式，不仅能将可训练参数再降一个数量级，还能利用 Lipschitz 边界防止微调过程中的路由坍缩。
2. **与 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) & `MerA` SVD 初始化的深刻同构**：
   * L2R 发现的“径向范数敏感性（Scale Sensitivity）”与我们在 *Transformer-Geometry* 及 `ads-rsi`（定律 ADS-RSI-1：Scale-Cancellation）中揭示的**“深层残差流径向范数 $\Vert h\Vert _ 2$ 掩盖切向语义方向 $h / \Vert h\Vert _ 2$ ”**完全一致！此外，在将稠密模型或预训练线性路由器 $W _ r \in \mathbb{R}^{N \times d}$ 转化为 L2R 路由器时，无需随机初始化 $P$ ，可直接调用我们的 **`MerA` 数据感知激活协方差 SVD（Activation-Covariance SVD）** 提取前 $r$ 个主奇异方向初始化 $P$ ，实现零冷启动抖动的低秩 Lipschitz 路由升级。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Low-Rank + SIPS Lipschitz-Stabilized MoD Router Head)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-27_ai_paper_notes.md`


---

### 3.10 [2026-09-26] 🔄 *LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language Modeling*
> **聚焦领域**：Looped Transformers · Mixture of Experts (MoE) · Iterative Depth Scaling · Weight Sharing  
> **arXiv**：[`arXiv:2606.04438`](https://arxiv.org/abs/2606.04438)

```
  输入表征 h^{(0)} ──► [ 循环步 t = 1..K : IterAdaLN(h, t) 轮次特征调制 ]
                                       │
                                       ▼
                     [ 共享 MoE 路由层: Top-k 稀疏专家激活 + 跨循环容量均衡 ]
                                       │
                                       ▼
                     [ 解耦总参数量 P 与单 Token 算力 FLOPs (同参数量 PPL 显著降低) ]
```

#### 🎯 背景与痛点剖析 (Problem Statement)
* **权重复用与轮次角色分化的矛盾**：在 Looped Transformer 中，直接将同一组 Transformer 块重复循环 $K$ 次，虽然能以 $O(1)$ 参数开销换取 $O(K)$ 的等效推理深度，但会导致两个严重退化：（1）不同循环步 $t \in \lbrace1, \dots, K\rbrace$ 缺乏步间身份区分，引发梯度震荡与隐状态平行分量 $\Delta h _ \parallel$ 爆炸；（2）若将循环架构直接与 MoE 结合，不同循环步会争抢同一批头部 Expert，导致严重的跨循环路由坍缩（Cross-Loop Routing Collapse）。

#### 💡 核心方法与底层数学实现 (Mathematical Formulations)
1. **迭代步自适应层归一化 (Iteration-Adaptive LayerNorm, `IterAdaLN`)**：
   - 为第 $t$ 次循环引入轻量级步间嵌入向量 $e _ t \in \mathbb{R}^d$ ，对共享主干的归一化层施加轮次特异性的仿射缩放与偏移调制：

$$
\text{IterAdaLN}(h^{(t)}, t) = \big(1 + \gamma(e _ t)\big) \odot \frac{h^{(t)} - \mu}{\sigma} + \beta(e _ t)
$$

   - 通过仅占总参数量 $<0.1$ % 的步间条件调制参数，赋予共享 MoE 块在不同循环深度下截然不同的几何变换角色。
2. **跨循环容量感知负载均衡 (Iteration-Aware Capacity Balancing)**：
   - 设第 $t$ 步第 $i$ 个专家的路由门控概率为 $p _ i^{(t)}(x)$ ，论文将辅助负载均衡损失扩展至循环时间轴与批次维度的联合分布上，防止特定专家在连续多次循环中被重复饱和激活。

#### 📊 关键实验与结论 (Experiments & Findings)
* **等参数量与等 FLOPs 双向碾压**：在语言建模基准与常识推理任务上，循环 $K=2\sim 4$ 步的 `LoopMoE` 在相同活跃参数量下显著优于标准稠密 Looped 模型，且在相同总参数预算下逼近非共享深层 MoE 模型的困惑度（PPL）上限。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作与在研主线**：
  * [Paper #16: *Disentangling Representation Evolution in Transformers through Directional Decomposition* (EMNLP 2026, `arXiv:2609.15975`)]
  * [Paper #11: *Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts* (ICLR 2026)]
  * [Paper #10: *Router-Tuning for Dynamic Mixture of Experts* (EMNLP 2025)]
  * [Active Line: *Physical AI / VLA-Loop (Stage-Wise Multi-LoRA Residual Boost & Adaptive Layer Looping)*]
* **🔬 机理对比与技术演进**：
  * `LoopMoE` 采用 `IterAdaLN`（逐通道对角缩放 $\gamma(e _ t)$ ）来区分不同循环轮次；而我们在 `VLA-Loop`（见 W39 研发笔记 9/22–9/23）中提出**用极小秩的 Stage-Wise LoRA 去编辑共享主干的每一次循环**，并进一步推进到了**逐层自适应决定是否 Loop**；
  * 从我们 *Transformer-Geometry (EMNLP 26)* 的正交方向分解视角来看，`IterAdaLN` 仅在归一化后施加坐标轴缩放，主要调节平行缩放分量 $\Delta h _ \parallel$ ；而我们的 **共享主干 + 轮次轻量 LoRA ( $\Delta W _ t = B _ t A _ t$ )** 则能直接在子空间中引入低秩正交旋转分量 $\Delta h _ \perp$ ，在表达能力上严格包含 `IterAdaLN`！
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 在撰写 `Physical AI` (MLSys) 论文的 Loop 章节时，可将 `LoopMoE` 的 `IterAdaLN` 作为轻量轮次调制的文献对照基准，用实验展示我们 **“共享主干 + MERA 初始化的轮次小 LoRA + 逐层自适应 Loop 路由”** 相比单纯 LayerNorm 调制的显著几何表达优势。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Dynamic Token Halting across Looped Transformer Steps)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-26_ai_paper_notes.md`


---

### 3.11 [2026-09-26] 🤖 *VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference*
> **聚焦领域**：Vision-Language-Action (VLA) · Embodied AI · Visual Token Pruning · Temporal Consistency  
> **arXiv**：[`arXiv:2511.16449`](https://arxiv.org/abs/2511.16449)

```
  连续控制帧视觉流 ──► [ 层级一 (Prefill): 跨模态指令-视觉语义重要度评估 ]
                                           │
                                           ▼
                       [ 层级二 (Decode): 时域指数平滑动作相关性追踪 S_t = λS_{t-1} + (1-λ)A_t ]
                                           │
                                           ▼
                       [ Combine-then-Filter 联合剪枝: 避免浅层误删关键操控锚点 ]
```

#### 🎯 背景与痛点剖析 (Problem Statement)
* **“语义显著性”与“动作控制必要性”的错位（Semantic-Action Gap）**：在机械臂精细操控任务（如 LIBERO）中，单帧静态视觉编码器认为显著的背景物体，未必是当前动作步（Action Chunk）夹爪需要接触的目标；反之，若在浅层仅凭静态视觉注意力盲目丢弃大量 Patch Token，会导致深层 Action Expert 丢失空间几何锚点，引发轨迹剧烈抖动。

#### 💡 核心方法与数学实现 (Mathematical Formulations)
1. **双层重要度融合准则 (Combine-then-Filter Dual-Level Criterion)**：
   - 同时提取语言指令在 Prefill 阶段对第 $i$ 个视觉 Token 的语义关注度 $I _ {\text{sem}}^{(i)}$ ，以及解码器生成动作 Token 时的交叉注意力得分 $I _ {\text{act}, t}^{(i)}$ ；
2. **跨时间步动作相关性平滑 (Temporal Action Smoothing)**：
   - 利用连续控制帧之间的时间连续性，引入历史动作注意力动量缓存：

$$
\tilde{I} _ {\text{act}, t}^{(i)} = \lambda \tilde{I} _ {\text{act}, t-1}^{(i)} + (1 - \lambda) I _ {\text{act}, t}^{(i)}
$$

   - 仅保留综合得分 $S _ t^{(i)} = I _ {\text{sem}}^{(i)} \cdot \tilde{I} _ {\text{act}, t}^{(i)}$ 最高的视觉 Token 子集。

#### 📊 关键实验与结论 (Experiments & Findings)
* 在 OpenVLA 与主流机器人操控基准（LIBERO-Spatial / Object / Goal / Long）上，剔除 **50%–75% 视觉 Token** 仍保持与全量 Token 持平的任务成功率，端到端控制频率显著提升。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作与在研主线**：
  * [Active Line: *Physical AI (`VLADrop` / `DTR` / `HiSTrim` Exclude-Self Value-Space Perp KV256)*]
  * [Paper #8: *Understanding and Harnessing Sparsity for Unified Multimodal Models* (TMLR 2026)]
  * [Paper #9: *Uncovering the Redundancy in Transformers via Layer Dropping* (TMLR 2025)]
* **🔬 机理对比与技术演进**：
  * 我们在 W38 周记（9/15–9/17）中深刻总结了两条核心定律：（1）**Layer 0（纯 ID Embedding、尚未经过上下文交互）绝不能直接做激进 Token Drop**，必须在表征充分上下文化之后再按浅层保守、深层激进的曲线压缩；（2）**VLA 的鲁棒性来源于三个时间尺度的“伤口愈合（Wound Healing）”纠错通道**（步内注意力、步间去噪、episode 内周期性视觉重锚）；
  * `VLA-Pruner` 的时域平滑动量 $\tilde{I} _ {\text{act}, t}$ 恰恰显式利用了我们指出的第三层“episode 内时域连续重锚”特性！
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 在 `Physical AI` (MLSys) 论文中，可将 `VLA-Pruner` 纳入 Related Work 与对比讨论，突出我们 **全栈四维协同压缩（数据 DTR + Token `HiSTrim` + 层 `VLADrop/Loop` + 步数 `SnapFlow` 单步蒸馏）** 相比单一视觉 Token 剪枝在真实硬件延迟（Batch=1 访存带宽瓶颈）上的系统级代差优势。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Dynamic Token Halting across Looped Transformer Steps)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-26_ai_paper_notes.md`


---

### 3.12 [2026-09-25] Fully Looped Transformer: Stabilizing Looped Models via Attention Injection and Residual Scaling

* **论文信息**：`arXiv:2605.18797` (2026-05)
* **核心关键词**：Fully Looped Transformer、Attention Injection、Anchor KV Grounding、Gradient Oscillation Prevention

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       Fully Looped Transformer with Parameter-Free Initial Attention Injection    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Initial Pass (k=0): Input Embedding H^{(0)} ---> Compute Anchor (K^{(0)}, V^{(0)})|
|                                        |                                          |
|                                        v                                          |
|  Loop Iteration k = 1 .. K:                                                       |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Anchor-Injected Multi-Head Attention (零参数初始锚点键值注入)            |  |
|  |    \tilde{K}^{(k)} = (1 - \lambda_k) K^{(k)} + \lambda_k K^{(0)}            |  |
|  |    \tilde{V}^{(k)} = (1 - \lambda_k) V^{(k)} + \lambda_k V^{(0)}            |  |
|  |    Prevents representation drift & provides direct gradient highway to k=0  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Unit-Sphere / Variance-Preserving Residual Update                        |  |
|  |    H^{(k+1)} = \text{Norm}\big( H^{(k)} + \frac{1}{\sqrt{K}} f_\theta(H^{(k)}, \tilde{K}^{(k)}, \tilde{V}^{(k)}) \big)|
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **深层循环中的“初始锚点遗忘”与反向传播雅可比谱半径失控**：当一个循环 Transformer 连续迭代 $K \ge 8$ 步时，第 $k$ 步的隐状态 $H^{(k)}$ 经过反复的非线性自注意力和 FFN 变换后，逐渐丢失了原始输入 Token 的精细词法锚点信息；同时在反向传播（BPTT）中，共享权重连乘 $\prod _ {k=1}^K \big(I + \frac{\partial f _ \theta}{\partial H^{(k)}}\big)$ 极易引发梯度震荡或消失。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **零参数初始注意力注入（Parameter-Free Attention Injection）**：
   缓存首轮（ $k=0$ ）计算得到的初始键值张量 $\left(K^{(0)}, V^{(0)}\right)$ 。在后续任意第 $k \in \lbrace1, \dots, K\rbrace$ 次循环中，通过凸组合或拼接将初始锚点注入当前步的注意力键值中：

$$
O^{(k)} = \text{Softmax}\left( \frac{Q^{(k)} \big( (1-\lambda) K^{(k)} + \lambda K^{(0)} \big)^\top}{\sqrt{d _ k}} \right) \Big( (1-\lambda) V^{(k)} + \lambda V^{(0)} \Big)
$$

   这一设计在计算图上为每一个循环步 $k$ 建立了一条直通初始表征 $\left(K^{(0)}, V^{(0)}\right)$ 的**一阶梯度短路高速通道（Direct Gradient Highway）**：

$$
\frac{\partial \mathcal{L}}{\partial H^{(0)}} = \frac{\partial \mathcal{L}}{\partial H^{(K)}} \prod _ {k=1}^K J _ k + \lambda \sum _ {k=1}^K \frac{\partial \mathcal{L}}{\partial O^{(k)}} \frac{\partial O^{(k)}}{\partial (K^{(0)}, V^{(0)})} \frac{\partial (K^{(0)}, V^{(0)})}{\partial H^{(0)}}
$$

   从而彻底消除了高循环步数下的梯度消失与震荡！

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在完全不增加任何额外参数（0 Extra Parameters）的条件下，Fully Looped Transformer 在 $K=8, 12$ 步循环预训练中完全消除了传统 Looped Transformer 的梯度尖峰（Gradient Spikes），验证集困惑度（PPL）降低 **`1.45`**，下游推理基准提升 **`+4.9%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **直接印证我们 `vla-loop` 定律（Lightweight Dropped-Span VLM Cross-KV Grounding）！**
  * 我们在 `vla-loop` 中发现，当动作专家循环迭代 $K=3,4$ 步时，若每一步都强绑回初始锚点 VLM Prefix KV（即此处的 $\left(K^{(0)}, V^{(0)}\right)$ ），即可完美阻止循环轨迹漂移！该论文的梯度短路公式为我们 `vla-loop` 的 Cross-KV Grounding 提供了极其漂亮的反向传播雅可比谱稳定性证明。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Dynamic Token Halting across Looped Transformer Steps)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-25_ai_paper_notes.md`


---

### 3.13 [2026-09-24] LearnPruner: Two-Stage Differentiable Visual Token Pruning for Large Vision-Language Models

* **论文信息**：`arXiv:2604.23950` (2026-04)
* **核心关键词**：Two-Stage Visual Token Pruning、Differentiable Gumbel/Sigmoid Masking、Shallow Deduplication & Deep Grounding

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       LearnPruner: Two-Stage Differentiable Visual Token Pruning for LVLMs        |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Visual Patch Tokens V^{(0)} (N_v = 576)                                          |
|          |                                                                        |
|          v                                                                        |
|  +-----------------------------------------------------------------------------+  |
|  | Stage 1 (Shallow Layer l_1): Vision-Intrinsic Redundancy Pruning            |  |
|  |    Removes background & spatially homogeneous patches BEFORE cross-modal    |  |
|  |    stabilizes -> Retains N_1 tokens                                         |  |
|  +-----------------------------------------------------------------------------+  |
|          |                                                                        |
|          v                                                                        |
|  +-----------------------------------------------------------------------------+  |
|  | Stage 2 (Mid Layer l_2): Instruction-Grounded Cross-Modal Pruning           |  |
|  |    Prunes task-irrelevant objects using stabilized text-to-vision attention |  |
|  |    Differentiable Soft-to-Hard Attention Bias: A_{i,j} + \log m_j(\tau)     |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **单阶段过早剪枝的“跨模态盲视”与过晚剪枝的“算力浪费”**：若在极浅层（如第 2 层）就仅凭文本指令去剪除大量视觉 Token，此时文本与视觉表征尚未完成跨模态对齐，极易误删目标物体；而若等到第 16 层才剪枝，前 16 层已经消耗了超过 50% 的全量视觉 FLOPs。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **浅层视觉内生去重 + 中层指令对齐聚焦的两阶段架构**：
   在浅层 $l _ 1$ ，仅基于视觉自注意力与空间局部方差剔除纯背景冗余块（保留率 $\rho _ 1 \approx 50$ %）；在中层 $l _ 2$ ，利用已对齐的跨模态交互特征进一步筛选与指令强相关的核心块（保留率 $\rho _ 2 \approx 15$ %）。
2. **注意力对数掩码软硬退火（Differentiable Log-Mask Annealing）**：
   训练期将连续重要性得分 $s _ j \in (0, 1)$ 通过温度 $\tau$ 转化为软掩码 $m _ j(\tau) = \sigma\big((s _ j - \theta _ {\text{thr}})/\tau\big)$ ，并以对数偏置注入注意力矩阵：

$$
\tilde{A} _ {i, j} = \frac{m _ j(\tau) \exp(q _ i^\top k _ j / \sqrt{d _ k})}{\sum _ {r} m _ r(\tau) \exp(q _ i^\top k _ r / \sqrt{d _ k})}
$$

   随着 $\tau \to 0^+$ ， $m _ j(\tau) \to \lbrace0, 1\rbrace$ ，训练期软注意力平滑收敛至推理期的物理硬剔除，实现零训练-推理鸿沟。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **LLaVA-1.5/NeXT** 与 **Qwen2-VL** 上，LearnPruner 仅保留 **11.1%–16.7% 视觉 Token**，FLOPs 降低 **68%**，在 10 项多模态基准上的平均精度达到全 Token 模型的 **99.6%**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Sparsity for Unified Multimodal Models* (TMLR 2026) & *Demystifying When Pruning Works via Representation Hierarchies* (ICML 2026) 的完美契合**：验证了根据表征层级演化阶段（浅层模态内去重 vs. 中层跨模态语义聚焦）分阶段设置不同剪枝准则的必要性。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (`CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-24_ai_paper_notes.md`


---

### 3.14 [2026-09-24] Training-Free Looped Transformers: Test-Time Mid-Stack Layer Looping

* **论文信息**：`arXiv:2605.23872` (2026-05)
* **核心关键词**：Training-Free Looped Transformer、Test-Time Depth Scaling、Mid-Stack Fixed-Point Iteration

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       Training-Free Looped Transformers: Test-Time Mid-Stack Layer Looping        |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Frozen Checkpoint: [Shallow Layers 1..l_a-1]                                     |
|                              |                                                    |
|                              v                                                    |
|        +---> [Mid-Stack Reasoning Span: Layers l_a .. l_b] ---+                   |
|        |                     |                                |                   |
|        |          Loop K times at Test Time                    |                   |
|        +--- Damped Contraction: h <- (1-\eta)h_{\text{in}} + \eta h_{\text{out}}  |
|                              |                                                    |
|                              v                                                    |
|                     [Deep Readout Layers l_b+1..L]                                |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **能否在不重新训练的情况下让现成开源大模型享受循环深度扩展？** 以往工作普遍认为 Looped Transformer 必须从头带循环拓扑预训练，否则直接把某一层重复执行会导致隐状态偏离后续层期望的输入流形。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **中段层块的近似压缩不动点迭代性质（Mid-Stack Contractive Mapping）**：
   作者分析发现，在预训练 Transformer 的中间深层区间 $[l _ a, l _ b]$ （通常位于 $0.4L \sim 0.75L$ ），相邻层的输入输出处于同一缓变语义流形上，复合块算子 $\mathcal{F} _ {l _ a:l _ b}$ 在局部切空间上近似构成压缩不动点精化映射。
2. **阻尼流形拉回循环更新（Damped Manifold-Preserving Loop）**：
   为防止在测试期重复调用 $\mathcal{F} _ {l _ a:l _ b}$ 时隐状态范数越界，在第 $k$ 次额外循环后施加范数匹配与阻尼凸组合：

$$
h^{(k)} = \frac{\Vert h^{(0)}\Vert _ 2}{\Vert\tilde{h}^{(k)}\Vert _ 2} \tilde{h}^{(k)}, \qquad \text{where } \tilde{h}^{(k)} = (1 - \eta) h^{(k-1)} + \eta \mathcal{F} _ {l _ a:l _ b}(h^{(k-1)})
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在完全零训练（Zero Finetuning）的 **Llama-3-8B** 与 **Mistral-7B** 上，对中段 6 层额外循环 $K=2$ 次，在 GSM8K、ARC-Challenge 与逻辑推理任务上直接获得 **`+2.1%` 至 `+3.8%`** 的免费准确率提升。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `vla-loop`（Layer-Specific Span-Bounded Dynamic Halting）及 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) 高度同源**：
  * 该文通过范数重缩放 $\frac{\Vert h^{(0)}\Vert _ 2}{\Vert\tilde{h}^{(k)}\Vert _ 2}$ 抑制测试期循环发散，本质上正是我们在 *Transformer-Geometry* 中指出的**抑制平行径向膨胀、仅保留球面切向正交精化**！

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Dynamic Token Halting across Looped Transformer Steps)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-24_ai_paper_notes.md`


---

### 3.15 [2026-09-23] StepKV: Step-Aware KV Cache Compression for Preserving Reasoning Continuity

* **论文信息**：`arXiv:2609.22158` (2026-09)
* **核心关键词**：Step-Aware KV Compression、Long-CoT Reasoning Continuity、Semantic Span Eviction、Discourse Boundary Detection

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       StepKV: Step-Aware KV Cache Compression for Long-CoT Reasoning Models       |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Generated CoT Stream ---> Boundary Detector (\n\n, "Wait,", equation delimiters) |
|                            Partitions tokens into Reasoning Steps {S_1, S_2..S_M} |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Step-Level Cohesive Saliency Scoring (推理步级内聚显著性打分)            |  |
|  |    U(S_m) = \frac{1}{|S_m|^\alpha} \sum_{j \in S_m} A_{\text{future} \to j} |  |
|  |             \cdot (1 + \gamma \cdot \mathbb{I}[\text{is\_milestone}(S_m)])  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Whole-Step Retention or Summary Compression (整步保留或边界锚点折叠)     |  |
|  |    Never punch holes inside an active mathematical equation or logic step!  |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **逐 Token 驱逐在长思维链（Long-CoT）中的“公式穿孔效应（Formula Swiss-Cheese Effect）”**：在 DeepSeek-R1 或 Qwen-QwQ 等推理模型生成数万 Token 的推导过程中，H2O/SnapKV 等按单 Token 注意力打分驱逐的方法往往会保留某一步方程的等号和首尾变量，却把括号内的中间符号剪掉。这种“半句话残骸”留在 KV 缓存中会严重误导后续注意力回溯，导致模型陷入重复验算死循环。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **推理步边界切分与步内完整性约束**：
   将长思维链序列划分为语义内聚的推理步集合 $\mathcal{S} = \lbrace S _ 1, S _ 2, \dots, S _ M\rbrace$ （以换行符、逻辑连接词或公式块定界）。引入步级二值保留决策变量 $z _ m \in \lbrace0, 1\rbrace$ （而非 Token 级独立变量），将预算为 $B$ 的 KV 缓存淘汰问题形式化为带步长权重的 0-1 背包优化：

$$
\max _ {z \in \lbrace0, 1\rbrace^M} \sum _ {m=1}^M z _ m \cdot \mathcal{U} _ {\text{step}}(S _ m) \qquad \text{s.t.} \quad \sum _ {m=1}^M z _ m |S _ m| + \sum _ {m=1}^M (1 - z _ m) c _ {\text{anchor}} \le B
$$

   其中被淘汰的冗余探索步（如已推翻的“Wait, let me recalculate”死胡同分支， $z _ m=0$ ）仅保留其末尾 $c _ {\text{anchor}}=2$ 个结论边界 Token 作为消极记忆锚点，而保留的关键推导步（ $z _ m=1$ ）则完整保留其内部全部 Token。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **AIME 2025**、**MATH-500** 与 **GPQA-Diamond** 上，StepKV 在压缩 **65%–75% CoT KV 缓存** 的条件下，相比逐 Token 驱逐的 SnapKV / H2O 将推理准确率大幅提升 **`+9.4%` 至 `+15.2%`**，并缩短了 18% 的无效重复反思长度。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *EffiR* (ACL 2026) 及 `Efficient Ads / HisTrim` 的块级结构化剪枝高度一致**：
  * 在我们的用户行为序列压缩（HisTrim）与长程推理压缩（EffiR）中，同样发现按完整事件/完整推理子句（Event/Step-Level）进行内聚度打分与整块保留，远比破坏局部语法结构的零散 Token 剔除稳健得多。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (`CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-23_ai_paper_notes.md`


---

### 3.16 [2026-09-23] HetDPT: Rethinking Depth Pruning for Vision Transformers — A Heterogeneity-Aware Perspective

* **论文信息**：`arXiv:2607.03784` (2026-07)
* **核心关键词**：Heterogeneity-Aware Depth Pruning、Decoupled MHSA/FFN Pruning、Vision Transformers

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|      HetDPT: Heterogeneity-Aware Decoupled Sub-Layer Depth Pruning for ViTs       |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Standard Block l:  X ---> [MHSA^{(l)} (Spatial Mixing)] ---> [FFN^{(l)} (Channel)]|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Sub-Layer Functional Heterogeneity Profiling (子层异构功能解耦剖析)      |  |
|  |    Deep MHSA layers exhibit high spatial attention map redundancy;          |  |
|  |    Shallow/Mid FFN layers exhibit higher channel transformation redundancy  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Independent Sub-Layer Pruning under Latency Constraint                   |  |
|  |    Can prune MHSA^{(l)} while keeping FFN^{(l)} (or vice versa) with zero   |  |
|  |    dimension mismatch via residual identity bypass                          |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **整块绑定剪枝（Coupled Block Pruning）忽略了注意力与 FFN 的深度角色错位**：传统深度剪枝总是将第 $l$ 层的 $\left( \text{MHSA}^{(l)}, \text{FFN}^{(l)} \right)$ 捆绑在一起同时保留或同时删除。然而在视觉与多模态编码器中，深层的空间跨 Token 交互（MHSA）早已收敛（注意力图趋于恒等或全局平均），但深层的逐 Token 特征非线性映射（FFN）仍在执行关键的语义分类投影。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **MHSA 与 FFN 异构解耦敏感度建模**：
   分别为每个子层引入独立的二值门控 $\left(m _ {\text{attn}}^{(l)}, m _ {\text{ffn}}^{(l)}\right) \in \lbrace0, 1\rbrace^2$ ：

$$
h _ {\text{mid}}^{(l)} = h^{(l-1)} + m _ {\text{attn}}^{(l)} \cdot \text{MHSA}^{(l)}\big(\text{LN} _ 1(h^{(l-1)})\big)
$$

$$
h^{(l)} = h _ {\text{mid}}^{(l)} + m _ {\text{ffn}}^{(l)} \cdot \text{FFN}^{(l)}\big(\text{LN} _ 2(h _ {\text{mid}}^{(l)})\big)
$$

   利用泰勒二阶敏感度联合硬件实测延迟表 $\tau _ {\text{attn}}, \tau _ {\text{ffn}}$ 求解整数线性规划（ILP）：

$$
\min _ {\lbrace m _ {\text{attn}}^{(l)}, m _ {\text{ffn}}^{(l)}\rbrace} \sum _ {l=1}^L \Big( (1 - m _ {\text{attn}}^{(l)}) \Omega _ {\text{attn}}^{(l)} + (1 - m _ {\text{ffn}}^{(l)}) \Omega _ {\text{ffn}}^{(l)} \Big) \quad \text{s.t.} \quad \sum _ {l=1}^L \big( m _ {\text{attn}}^{(l)} \tau _ {\text{attn}} + m _ {\text{ffn}}^{(l)} \tau _ {\text{ffn}} \big) \le T _ {\text{budget}}
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **DeiT**、**Swin** 与 **CLIP-ViT-L/14** 上，HetDPT 在相同 **1.5x–1.8x 硬件实测加速比** 下，比整块深度剪枝提升了 **`+1.9%` 至 `+3.2%`** 的 ImageNet 与多模态下游准确率。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Layer Dropping* (TMLR 2025) & `vla-dtr` 的子层解耦路由完美呼应**：在 VLA 视觉主干与动作专家的深度剪枝中，深层 Cross-Attention 往往比 FFN 更早饱和，采用解耦子层跳过可进一步压榨 15% 延迟。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Decoupled Attention-MoD vs FFN-MoD Routing Budgets)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-23_ai_paper_notes.md`


---

### 3.17 [2026-09-23] HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist Vision-Language-Action Policies

* **论文信息**：`arXiv:2512.05693` (2025/2026)
* **核心关键词**：Hierarchical MoE、Generalist VLA Policy、Task-Skill Decoupled Routing、Gradient Conflict Mitigation

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       HiMoE-VLA: Hierarchical Mixture-of-Experts for Generalist VLA Policies      |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Language Goal + Visual State ---> [Level-1: Task/Embodiment Router G_{\text{task}}]|
|                                           |                                       |
|                   Selects Domain Expert Group \mathcal{G}_m                       |
|                                           v                                       |
|         Proprioception + Local Patch ---> [Level-2: Skill Primitive Router G_{\text{skill}}]|
|                                           |                                       |
|                   Activates Fine-Grained Motor Primitives (Reach / Grasp / Place) |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **异构本体与多任务联合训练中的“扁平路由混淆”**：在跨机械臂本体、跨数十种操作任务的通用 VLA 训练中，单层扁平 MoE 路由器容易按表层视觉背景而非底层运动学技能聚类，导致不同任务间出现严重的负迁移。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **双层语义-运动解耦条件路由（Bi-Level Semantic-Kinematic Conditional Routing）**：
   高层路由器 $G _ {\text{task}}(c _ {\text{lang}}, I _ {\text{global}})$ 根据语言指令与全局视觉场景选择任务簇 $m \in \lbrace1, \dots, M\rbrace$ ，低层路由器 $G _ {\text{skill}}^{(m)}(s _ {\text{prop}}, I _ {\text{wrist}})$ 根据本体关节状态与腕部相机高频特征在簇内选择动作基元专家 $e \in \mathcal{E} _ m$ ：

$$
P(e \mid x) = \sum _ {m=1}^M G _ {\text{task}}(m \mid c _ {\text{lang}}, I _ {\text{global}}) \cdot G _ {\text{skill}}^{(m)}(e \mid s _ {\text{prop}}, I _ {\text{wrist}}) \cdot \mathbb{I}(e \in \mathcal{E} _ m)
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在跨 50+ 任务的 Open-X Embodiment 与仿真套件上，HiMoE-VLA 比同激活参数量的稠密 VLA 与单层 MoE-VLA 平均成功率提升 **`+8.7%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `ads-rsi` 中的 GemTagger 分层路由与 *Router-Tuning* (EMNLP 2025) 高度契合**：将高层任务上下文路由与底层高频状态路由树状解耦，可大幅提升细粒度专家的专业化纯度。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (`CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-23_ai_paper_notes.md`


---

### 3.18 [2026-09-23] D-Cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding

* **论文信息**：`arXiv:2607.14647` (2026-07)
* **核心关键词**：Speculative Decoding、Verification Depth Pruning、Cross-Request Budget Allocation

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       D-Cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Batched Draft Trees {T_1, ..., T_B} with Draft Confidence Scores {c_1, ..., c_B} |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Early-Layer Margin Verification (浅层置信度提前决断)                        |  |
|  |    At Intermediate Layer L_{\text{cut}} < L:                                |  |
|  |    If early logit margin \Delta z^{(L_{\text{cut}})} >> \tau_accept or << -\tau_reject:|
|  |    Drop verified/rejected draft tokens from remaining layers L_{\text{cut}}+1..L|
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **批量投机解码验证阶段的深层算力浪费**：在大 Batch 投机解码中，目标大模型需要同时并行验证每个请求的 $K$ 个草稿 Token。实际上，超过 70% 的简单正确草稿或明显错误的草稿在目标模型的前 60% 层就已经毫无悬念地分出胜负，继续让它们跑完后 40% 层纯属浪费。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **基于中间层 Logit 间隔的动态截断准则**：
   在中间探测层 $l _ {\text{probe}}$ ，通过轻量早期退出投影计算草稿 Token $y _ i$ 的对数概率边际 $\Delta _ {i}^{(l)} = \hat{\ell}^{(l)}(y _ i) - \max _ {v \neq y _ i} \hat{\ell}^{(l)}(v)$ 。当 $|\Delta _ {i}^{(l)}| > \gamma _ l$ 时，立即锁定接受/拒绝决策，并将该草稿及其后续依赖子树从第 $l+1 \dots L$ 层的批次张量中动态压缩移除。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 Batch Size = 16–64 的生产级投机解码服务中，D-Cut 将验证阶段算力开销削减 **38%**，端到端吞吐在 EAGLE-2 基线上进一步提升 **1.42x**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Capacity-Aware Inference* (ICLR 2026) & *Layer Dropping* (TMLR 2025) 直接协同**：在多步推理或投机验证中引入中间层间隔早退门控，可显著提升高并发批次下的有效吞吐。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Confidence-Margin Early Depth Termination)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-23_ai_paper_notes.md`


---

### 3.19 [2026-09-21] DeepLoop: Depth Scaling for Looped Transformers

* **论文信息**：`arXiv:2607.13491` (2026-07)
* **核心关键词**：Looped Transformers、Residual-Scaling Problem、Coherent Variance Growth、Depth Scaling Law

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|               DeepLoop: Depth Scaling for Looped Transformers                     |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Unrolled Standard Transformer (Independent Weights W_l):                         |
|    \text{Var}(h^{(L)}) \approx \text{Var}(h^{(0)}) + \sum_{l=1}^L \sigma_l^2 = O(L)|
|                                                                                   |
|  Naive Looped Transformer (Shared Weight W reused K times):                       |
|    Coherent alignment \langle f_W(h^{(k)}), f_W(h^{(j)}) \rangle > 0              |
|    ===> \text{Var}(h^{(K)}) = O(K^2)  [Catastrophic Residual & Gradient Explosion]|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | DeepLoop Coherent-Aware Residual Scaling & Step-Conditioned Norm            |  |
|  |    h^{(k)} = h^{(k-1)} + \frac{\alpha_k}{K^{\gamma}} f_W\big(\text{LN}_k(h^{(k-1)})\big)|
|  |    where \gamma \in [1/2, 1] interpolates between diffusive & coherent drift|  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **循环复用的“相干方差爆炸（Coherent Variance Explosion）”**：在标准非循环 Transformer（如 DeepNorm / Pre-LN）中，由于各层权重 $W^{(l)}$ 相互独立，层间残差增量的交叉协方差近似为零，因此 $L$ 层后的隐状态方差按随机游走以 $O(L)$ 线性增长（仅需 $1/\sqrt{L}$ 缩放）。然而在 **Looped Transformer** 中，同一物理层 $f _ W$ 被连续迭代调用 $K$ 次，第 $k$ 步的残差增量 $f _ W(h^{(k-1)})$ 与前一步高度正相关（相干叠加），导致隐状态范数以 ** $O(K^2)$ 二次方速度爆炸**，使得循环步数 $K > 4$ 时训练迅速崩溃！

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **相干循环残差方差增长定理（Coherent Residual Variance Theorem）**：
   设循环块映射为 $h^{(k)} = h^{(k-1)} + \beta _ k f _ W(h^{(k-1)})$ 。令步间余弦相关系数为 $\rho _ {j,k} = \frac{\mathbb{E}[\langle f _ W(h^{(j)}), f _ W(h^{(k)}) \rangle]}{\Vert f _ W(h^{(j)})\Vert _ 2 \Vert f _ W(h^{(k)})\Vert _ 2}$ 。当 $\rho _ {j,k} \ge \bar{\rho} > 0$ 时， $K$ 步循环后的终端方差满足：

$$
\mathbb{E}\big[\Vert h^{(K)} - h^{(0)}\Vert _ 2^2\big] = \sum _ {k=1}^K \beta _ k^2 \sigma _ f^2 + 2 \sum _ {1 \le j < k \le K} \beta _ j \beta _ k \rho _ {j,k} \sigma _ f^2 = \Theta\left( \Big(\sum _ {k=1}^K \beta _ k\Big)^2 \right)
$$

2. **DeepLoop 步间解耦缩放法则（Coherence-Compensated Scaling Law）**：
   为保证无论循环深度 $K$ 如何扩展，终端隐状态流形半径始终保持 $\Theta(1)$ 李雅普诺夫有界，DeepLoop 引入经验相干指数 $\gamma(\bar{\rho}) = \frac{1}{2} + \frac{1}{2}\bar{\rho} \in [\frac{1}{2}, 1]$ ，设定第 $k$ 步残差门控缩放系数为：

$$
\beta _ k(K) = \frac{c _ k}{K^{\gamma(\bar{\rho})}}, \qquad \text{with step-specific affine gain } \text{LN} _ k(h) = \gamma _ k \odot \frac{h - \mu}{\sigma} + b _ k
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在循环深度从 $K=2$ 扩展至 ** $K=16$ ** 的语言与数学推理预训练中，标准 Pre-LN 循环架构在 $K \ge 6$ 时完全发散，而 **DeepLoop** 稳定收敛并实现随循环次数 $K$ 对数线性下降的测试集 Loss，以 **1/4 的物理参数量** 追平同有效深度标准 Transformer 的推理性能。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **为我们 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) 与 `vla-loop`（定律 v18：Continuous Horizon-Phase Terminal Decay）提供精确的二阶统计力学解释！**
  * DeepLoop 发现的“相干叠加 $\rho _ {j,k} > 0$ 导致 $O(K^2)$ 范数爆炸”，从几何上看正是因为共享权重 $f _ W$ 在每次循环中持续向**平行径向分量 $\Delta h _ \parallel$ ** 注入同向推力！这再次证明了我们在 `vla-loop` 与 *Transformer-Geometry* 中剔除平行分量、仅保留正交切空间更新 $\Delta h _ \perp$ （使 $\rho _ {j,k}^{\parallel} \to 0$ ，从而将方差增长压回良性的 $O(K)$ ）并配合终端步长衰减 $\left(1-\tau _ k\right)^\beta$ 的根本必要性。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Dynamic Token Halting across Looped Transformer Steps)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-21_ai_paper_notes.md`


---

### 3.20 [2026-09-21] RotateK: Rotation-Aligned Key Channel Pruning for Vision-Language Models

* **论文信息**：`arXiv:2605.19218` (2026-05)
* **核心关键词**：Key Channel Pruning、Orthogonal Rotation Alignment、Vision-Language Models (VLMs)、Head-Dimension Compression

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       RotateK: Rotation-Aligned Key Channel Pruning for Vision-Language Models    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Attention Score Invariance under Orthogonal Rotation R \in O(d_k):               |
|    Q K^\top = (Q R)(K R)^\top   where R^\top R = I_{d_k}                          |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Cross-Modal Key-Query Co-Energy SVD (跨模态查询-键联合能量奇异值对齐)    |  |
|  |    Compute covariance C_K = \mathbb{E}[K_{\text{vis}}^\top K_{\text{vis}}]  |  |
|  |    Eigendecompose C_K = R \Lambda R^\top ---> Fold R into W_Q, W_K offline  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Tail Channel Truncation (尾部低能量通道截断: 兼容 RoPE 2x2 块旋转)       |  |
|  |    Retain top-r channels (r = 0.4 d_k) -> 60% Key Cache & GEMM Reduction    |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **原始坐标轴下的通道能量弥散**：在多模态大模型（VLM）中，除序列长度方向（Token 维度）冗余外，注意力头内部的特征维度 $d _ k$ （如 $d _ k=128$ ）在视觉特征空间中实际上具有极低的本征秩。然而，在原始训练得到的正交基下，信号能量均匀弥散在全部 128 个通道上，直接按坐标轴剪除任何通道都会造成较大的内积误差 $\Vert Q K^\top - \tilde{Q} \tilde{K}^\top\Vert _ F$ 。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **RoPE 兼容的分块正交旋转能量集中（RoPE-Compatible Block-Orthogonal Rotation）**：
   由于旋转位置编码（RoPE）以二维子平面 $\left(2i, 2i+1\right)$ 为单位作用： $R _ \Theta(m) = \text{diag}(R _ {\theta _ 1}^{(m)}, \dots, R _ {\theta _ {d _ k/2}}^{(m)})$ ，为保持与 RoPE 的可交换性，RotateK 将 $d _ k/2$ 个二维频率对按预期内积能量贡献 $\mathcal{E} _ i = \mathbb{E}\big[ \Vert q _ {[2i:2i+1]} \Vert _ 2^2 \cdot \Vert k _ {[2i:2i+1]} \Vert _ 2^2 \big]$ 进行重排，并在每个同频子空间内执行正交主轴对齐 $U _ i \in O(2)$ ：

$$
\tilde{W} _ Q = W _ Q U _ {\text{rot}}, \qquad \tilde{W} _ K = W _ K U _ {\text{rot}}
$$

2. **误差上界最小化通道截断**：
   保留能量最高的前 $r$ 个通道子块，此时注意力 logit 截断误差满足紧上界：

$$
\mathbb{E}\big[ | q^\top k - \tilde{q} _ {1:r}^\top \tilde{k} _ {1:r} |^2 \big] \le \sum _ {i = r/2 + 1}^{d _ k/2} \lambda _ i(C _ Q) \lambda _ i(C _ K)
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **LLaVA-NeXT**、**Qwen2-VL-7B** 与 **InternVL-2** 上，RotateK 剪除 **50%–60% 的 Key 通道**而无需微调，且与视觉 Token 剪枝（如 FastV / VLA-Pruner）**100% 正交兼容**，联合实现 **4.2x** 注意力加速且 VQA 精度损失 `<0.5%`。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `MerA` SVD 初始化及 *Sparsity for Unified Multimodal Models* (TMLR 2026) 的正交协同**：
  * RotateK 在特征通道维度 $d _ k$ 上的正交旋转浓缩与我们在 Token 维度 $N _ {\text{vis}}$ 上的剪枝构成了完整的二维矩阵联合低秩逼近（Row + Column Dual Sparsity），可直接嵌入 `vla-distillation` 的视觉前缀压缩器中。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (`CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-21_ai_paper_notes.md`


---

### 3.21 [2026-09-21] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection

* **论文信息**：`arXiv:2602.03216` (2026-02)
* **核心关键词**：Token Sparse Attention、Interleaved Compress-Decompress、Reversible Token Selection、Dense Kernel Compatibility

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|     Token Sparse Attention (TSA): Interleaved Reversible Token Sparsification     |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Layer l Input Hidden States H^{(l)} \in R^{L x d}                                |
|          |                                                                        |
|          +---> [Select Top-M Active Tokens I_l] ---> Gather Q_sub, K_sub, V_sub   |
|          |                                                   |                    |
|          |                                                   v                    |
|          |                                      Dense FlashAttention (M x M)      |
|          |                                                   |                    |
|          +---> [Scatter-Add Back to Full Length L] <---------+                    |
|          |                                                                        |
|          v                                                                        |
|  Layer l+1 Input H^{(l+1)} \in R^{L x d} (Previously skipped tokens can re-awake!)|
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **永久性 Token 丢弃（Permanent Token Dropping）的不可逆信息损失**：传统早退或逐层漏斗式 Token 剪枝（如 FastV、PyramidDrop）一旦在第 $l$ 层将某个 Token 丢弃，该 Token 在后续第 $l+1 \dots L$ 层中便永远消失。然而，在多跳推理或长文档问答中，浅层看似不相关的背景段落往往需要在深层推理出中间结论后才被重新检索激活。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **层内 Gather-Attention-Scatter 可逆稀疏算子**：
   在第 $l$ 层，轻量路由器根据当前隐状态打分选出活跃下标集 $\mathcal{I} _ l \subset \lbrace1, \dots, L\rbrace$ （ $|\mathcal{I} _ l| = M = \rho L \ll L$ ）。通过行抽取算子 $P _ {\mathcal{I} _ l} \in \lbrace0, 1\rbrace^{M \times L}$ 构造紧凑子矩阵：

$$
\tilde{Q} = P _ {\mathcal{I} _ l} Q, \quad \tilde{K} = P _ {\mathcal{I} _ l} K, \quad \tilde{V} = P _ {\mathcal{I} _ l} V \in \mathbb{R}^{M \times d}
$$

   在紧凑稠密张量上直接调用标准 FlashAttention-3 内核计算 $\tilde{O} = \text{FlashAttn}(\tilde{Q}, \tilde{K}, \tilde{V})$ ，随后通过转置散射算子 $P _ {\mathcal{I} _ l}^\top$ 还原回全序列残差流：

$$
H^{(l+1)} = H^{(l)} + P _ {\mathcal{I} _ l}^\top \big( \tilde{O} W _ O \big)
$$

   由于非活跃 Token $j \notin \mathcal{I} _ l$ 通过恒等残差分支完整保留了其隐状态 $H _ j^{(l)}$ ，它在第 $l+1$ 层可根据更新后的全局语义被重新选入 $\mathcal{I} _ {l+1}$ ！

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 64K–128K 多跳检索与大海捞针基准（RULER Multi-Hop Tracing）上，不可逆 Token 剪枝在 70% 稀疏度下准确率跌至 `31.2%`，而 **Token Sparse Attention** 保持了 **`88.4%`** 的高准确率，同时因完全复用稠密 FlashAttention 内核实现了 **2.6x** 真实注意力加速。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Demystifying When Pruning Works via Representation Hierarchies* (ICML 2026) & *Layer Dropping* (TMLR 2025) 的本质联系**：
  * TSA 的 `Gather -> Attention -> Scatter-Add` 本质上是对非活跃 Token 执行了**“Token 级条件层跳过（Token-Wise Conditional Layer Dropping）”**！这为我们把整层跳过（Layer Dropping）细粒度化为每个循环步/每层的动态子集更新提供了极佳的硬件友好范式。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Reversible Token Bypass vs Irreversible Drop in MoD)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-21_ai_paper_notes.md`


---

### 3.22 [2026-09-20] CARE: Spend Experts Where You Are Unsure — Confidence-Adaptive Routing for MoE-LoRA

* **论文信息**：`arXiv:2607.26052` (2026-07)
* **核心关键词**：Confidence-Adaptive Routing、MoE-LoRA、Nucleus Expert Activation、Router Uncertainty Entropy

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|            CARE: Confidence-Adaptive Routing for Mixture-of-Experts               |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Token Hidden State h_t ---> Router Probabilities p_t = Softmax(W_r h_t) \in \Delta^E|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Router Uncertainty Quantification (路由分布置信度/不确定性度量)          |  |
|  |    Sort probabilities: p_{t,(1)} >= p_{t,(2)} >= ... >= p_{t,(E)}           |  |
|  |    High confidence (peaked p_t) -> K_t = 1; High entropy -> K_t = K_{\max}  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Nucleus & Margin-Gated Dynamic Top-K(t) Selection                        |  |
|  |    K_t = \min \{ k \in [K_{\min}, K_{\max}] : \sum_{i=1}^k p_{t,(i)} >= \tau_p|
|  |               \text{ or } p_{t,(k)} - p_{t,(k+1)} >= \tau_m \}              |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **静态 Top- $k$ 路由的算力错配**：标准 MoE 对序列中的每一个 Token（无论是标点符号、常见停用词，还是复杂的逻辑转折词）均无差别地激活固定数量 $k$ 个专家。对于路由器高度确信的简单 Token（例如 $p _ {t,(1)} > 0.85$ ），强制拉起第 $2 \dots k$ 个低概率专家不仅浪费算力，还会引入长尾噪声干扰；而对于处于知识边界的模糊 Token，固定 $k$ 个专家又不足以覆盖多维语义假设。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **累积概率核与边际跳变双门控（Nucleus & Margin Gated Dynamic $K _ t$ ）**：
   将排序后的专家门控概率记为 $p _ {t,(1)} \ge p _ {t,(2)} \ge \dots \ge p _ {t,(E)}$ 。CARE 为每个 Token $t$ 动态分配激活专家个数 $K _ t \in [K _ {\min}, K _ {\max}]$ ：

$$
K _ t = \min \left\lbrace k \in \lbrace K _ {\min}, \dots, K _ {\max}\rbrace \middle| \sum _ {i=1}^k p _ {t,(i)} \ge \tau _ {\text{nuc}} \lor \big(p _ {t,(k)} - p _ {t,(k+1)}\big) \ge \tau _ {\text{margin}} \right\rbrace
$$

2. **零训练即插即用温度校准（Temperature Calibration under Global FLOPs Target）**：
   给定目标平均激活专家预算 $\bar{K} _ {\text{target}}$ ，在校准集上通过单标量温度 $\beta$ 缩放路由 logits $p _ t(\beta) = \text{Softmax}(W _ r h _ t / \beta)$ ，满足 $\mathbb{E} _ t[K _ t(\beta)] = \bar{K} _ {\text{target}}$ 。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在多任务 MoE-LoRA 与稀疏 MoE 语言模型上，CARE 在削减 **32%–45% 平均专家激活 FLOPs** 的同时，在常识推理、代码与数学基准上全面持平甚至超越固定 Top- $k$ 基线（`+0.9%` 平均准确率）。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Capacity-Aware Inference* (ICLR 2026) & *Router-Tuning* (EMNLP 2025) 的协同**：可将 CARE 的 Token 级置信度核门控（Nucleus Routing）与我们在 ICLR 2026 中提出的硬件容量感知丢弃/重路由（Capacity-Aware Dropping）级联，在软件置信度与硬件队列容量两个维度同时实现最优分配。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (Cumulative Probability Nucleus + Marginal Jump Dynamic Depth Gate)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-20_ai_paper_notes.md`


---

### 3.23 [2026-09-18] 🧩 *MoE-Tile: Warp-Aligned Tensor Slicing for Zero-Overhead Dynamic Sparse Routing on Modern Accelerators*
> **聚焦领域**：Mixture of Experts (MoE) · GPU Kernel Optimization · Warp Divergence · Hardware-Aware Sparsity  
> **arXiv**：[`arXiv:2609.09112`](https://arxiv.org/abs/2609.09112)

```
  动态 Token 路由序列 ──► [ 块级非对齐碎片 (Warp 严重分化) ] ──► 计算利用率 38%
                                      │
                                      ▼
                      [ MoE-Tile 算子: Warp-Aligned 2D Slicing ]
                                      ├── 线程块内 128x128 Tile 对齐填充
                                      └── 零开销 TMA (Tensor Memory Accelerator) 异步流水
                                      ▼
                    [ 算子级 GEMM 吞吐达 89% 理论峰值 (提速 2.34x) ]
```

#### 🎯 背景与硬件级痛点剖析 (Hardware Bottleneck)
* **动态门控与 GPU 线程块的天然矛盾**：MoE 模型的 Top- $k$ 门控路由将不同数量的 Token 动态分发给不同 Expert。在 GPU 底层执行专家 FFN 矩阵乘（GEMM）时，每个 Expert 分配到的实际 Token 数（Batch $M _ e$ ）并非硬件友好的 128 或 256 的倍数，导致大量 Warp 处于空转分化（Warp Divergence）状态，且引发非连续非对齐的显存搬运（Uncoalesced Memory Access），Tensor Core 实际利用率极低。

#### 💡 核心方法与原文底层工程实现 (Detailed System Mechanism)
1. **Warp 对齐二维分块调度器 (Warp-Aligned 2D Tile Slicer)**：
   - 设第 $e$ 个 Expert 接收到的 Token 数量为 $M _ e$ ，隐藏维度为 $K$ 与 $N$ ；
   - 传统实现采用 Padding 将 $M _ e$ 补齐到固定上界（造成显存与计算浪费），或采用 Ragged Batch（引发线程分化）；
   - 原文提出跨 Expert 全局排队与 Tile 重映射机制：将所有专家的计算任务切分为固定大小的硬件微块 $\mathcal{T} _ {i,j} \in \mathbb{R}^{128 \times 128}$ ，将跨 Expert 的边界碎片（Tail Residuals）打包组合进统一的共享微块中执行。
2. **硬件 TMA 异步流水线重叠 (Asynchronous TMA Pipelining)**：
   - 利用现代 GPU（Hopper/Blackwell）的 Tensor Memory Accelerator（TMA），在 Shared Memory 与 Global Memory 之间构建三级流水线缓冲，将 Tile 索引寻址重排序开销完全隐藏在 FFN 计算延迟内部。

#### 📊 关键实验与结论 (Experiments & Findings)
* **硬件测试平台**：NVIDIA H100 80GB SXM5 与 B200 GPU 集群；
* **测试模型**：DeepSeek-V2/V3 (236B/671B)、Mixtral-8x22B；
* **实测性能**：
  * **算子级 GEMM 计算吞吐**：相比标准 Megatron-LM 与 vLLM MoE 算子，计算吞吐提升 **2.34 倍**，Tensor Core 利用率从 38.2% 提升至 **89.1%**；
  * **端到端端 Decode 延迟**：Token 生成阶段延迟降低 **43.5%**，完全消除了动态路由带来的硬件抖动。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作**：
  * [Paper #11: *Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts* (ICLR 2026)]
  * [Paper #10: *Router-Tuning for Dynamic Mixture of Experts* (EMNLP 2025)]
  * [Paper #5: *MEO: Mixture of Experts Optimization* (EMNLP 2023)]
* **🔬 机理对比与技术演进**：
  * 我们在 *Capacity-Aware Inference (ICLR 26)* 中从**分布式全局宏观调度层面**定义了动态 Capacity Factor 与 Token 溢出分配策略；
  * *MoE-Tile* 则在**单卡底层 CUDA 算子与微架构 Tile 粒度**上解决了非规整 Token 批处理的执行开销；
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 可将我们 ICLR 26 的全局分布式调度器与 MoE-Tile 底层 Triton/CUDA 算子进行纵向打通：由我们算法在上层输出动态均衡的 Expert 负载约束，下层由 MoE-Tile 执行 128 对齐的极速计算，构建从分布式集群到单卡底层内核的端到端超高效 MoE 推理栈。

---

> [!TIP]
> **🎯 `Router-Tuning-Mixture-of-Depths` 仓库代码级落地点 (`Target Module`)**：`router_tuning/` (`CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-18_ai_paper_notes.md`


---
