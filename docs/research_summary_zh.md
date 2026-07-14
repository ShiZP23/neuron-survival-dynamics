# 神经元生存、结构相与初始化动力学研究总结

## 1. 研究主线

本项目最初研究 survival-based prune/regrow 是否能在保持任务性能的同时形成更有效的网络结构，随后逐步转向一个更基础的问题：在相同任务和训练规则下，不同随机路径为什么会进入不同的结构相，这些相如何影响层间功能组织、最佳性能与后期保持能力。

## 2. 方法演进

### 阶段一：生存式剪枝与再生

- 用 `mean(abs(activation)) × outgoing-weight norm` 定义当前重要性。
- 用 EMA 跟踪神经元的长期重要性。
- 仅对层内低 EMA 候选执行 validation ablation。
- 只有当移除神经元引起的损失增长低于阈值时才真正删除。
- 支持 fixed、prune-only、random regrow 和 strong-neuron split regrow。
- 独立控制 data、model initialization、shuffle 和 structure RNG。

### 阶段二：结构相分化

在 hard 合成回归任务上，同一规则重复出现 `prune0_only`、`prune0+2` 和少量 `no_prune` 相。分析从最终误差扩展到 phase commit、active share、best-to-final degradation 与 post-best drift。

### 阶段三：无剪枝对照与 shadow pruning

通过无显式结构更新的普通网络和 shadow pruning，判断结构相是否只由剪枝创造，还是在稠密训练中已经以潜在形式存在。

### 阶段四：Init-only Lottery-Ticket 式研究

固定数据和训练随机性，只改变初始化；在 MNIST、Fashion-MNIST 和 CIFAR-10 上比较 fixed、prune-only 和 fixed+shadow，建立 coarse/fine phase taxonomy、early marker、threshold sign 和 mismatch 分析。

### 阶段五：定向干预

对 boundary seed 和 timing-gap seed 进行指定 epoch、指定层的真实剪枝干预，以区分相关性预测和结构反馈的因果改写。

## 3. 主要结论

1. **结构相可重复存在。** 同一任务和规则不会收敛到唯一结构，不同初始化或数据路径会进入不同层间组织。
2. **结构承诺早于性能最优。** phase commit 往往发生在很早的结构更新，而最佳 checkpoint 出现在训练后期。
3. **潜力与保持能力分离。** `prune0+2` 相可达到略好的 best loss，却具有显著更大的 best-to-final degradation。
4. **相位是功能组织而非剪枝标签。** 不同相位对应长期稳定的 layer-wise active-share 分工。
5. **多数结构倾向由初始化诱导。** 40-seed CIFAR-10 cohort 中，shadow coarse/fine match rate 分别为 87.5% 和 82.5%。
6. **真实结构反馈主要发生在边界样本。** 40 个 seed 中仅 7 个 mismatch；定向干预表明这些 mismatch 包含多种 family-specific feedback route。
7. **Timing gap 不是单一机制。** 已观察到 distributed upstream trigger 与 layer-0-coupled timing trigger，并可能伴随不同 side effects。
8. **剪枝更像结构相的放大镜。** 当前证据提示潜在结构分化并非剪枝系统独有，但仍需在标准无剪枝网络上进一步验证。

## 4. 40-seed cohort

当前 CIFAR-10 SmallConv init-only cohort 包含：

- 40 个初始化 seed；
- 8 类 coarse phase；
- 18 类 fine phase；
- 7/40 个 mismatch seed；
- 多个 absence-type、timing-gap 和 second-update boundary family。

权威结果索引位于 `results/init_only_lth_20260401/README.md`，但全量 `results/` 为本地生成数据，不纳入版本控制。GitHub 中保留生成这些结果的源码、实验入口、研究协议、精选 manifest 和 study README。

## 5. 可复现资产

- 核心训练代码：`src/neuron_survival_dynamics/`
- init-only 模型与训练：`src/neuron_survival_dynamics/init_only/`
- 批量运行和报告脚本：`scripts/`
- 评估协议：`docs/evaluation_protocol.md`
- 研究目录设计：`docs/research_layout.md`
- 各专题研究说明：`studies/`
- 精选结果清单：`curated_results/`

## 6. 下一步

- 在标准 MLP/CNN 上直接测试 seed-dependent latent phase。
- 用 validation-aware early stopping 和 checkpoint freeze 分离泛化上限与 retention。
- 对关键 family 扩大 seed 数并预注册 intervention 规则。
- 将 phase taxonomy、threshold persistence 和 intervention evidence 收敛为论文主图。
- 发布最小复现数据包，而不是上传全量原始运行目录。
