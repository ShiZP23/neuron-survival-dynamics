# 标准不剪枝网络的潜在相分化研究

## 研究问题

我们已经在 `prune_grow_split` 中看到明显的结构相分化，也在 `prune_only` 中看到 seed 会把运行带到不同的剪枝模式。这个 study 的核心问题不是再证明一次“剪枝网络会分化”，而是回答下面三个更强的问题：

1. 标准 `fixed` 不剪枝网络里，是否已经存在与这些 `prune_only` 相标签对应的潜在分化？
2. `prune_only` 观察到的 `prune0+2` / `prune0_only` / `no_prune`，到底是在揭示 dense 网络本来就有的训练路径，还是在结构编辑后才被制造出来？
3. 如果把 `fixed` 运行按“同 seed 的 `prune_only` 相标签”来重排，不同组之间会不会已经在后期稳定性、功能重分配和泛化 gap 上分开？

## 核心策略

这里不先对 `fixed` 做无监督聚类，而是先采用一个更强、更可解释的外部索引：

- 对每个 `hard` 任务 seed，先看同 seed 的 `prune_only` 运行最终落在哪个 phase。
- 再把对应的 `fixed` 运行按这个 phase 标签重排。
- 如果 `fixed` 在这种外部标签下已经分开，说明剪枝更像是把潜在分化显化出来。
- 如果 `fixed` 完全不分开，而 `prune_only` 明显分开，说明剪枝本身更可能是主要分化驱动。

这比直接比较 `fixed` 和 `prune_only` 的均值更好，因为它保留了 seed-level 的路径信息。

## 当前输入数据

主分析池：

- `results/publishable_pilot_20260313/hard/prune_only`
- `results/publishable_pilot_20260313/hard/fixed`

平衡锚点池：

- `results/followup_20260313/hard/prune_only`
- 对应 seed 的 `results/publishable_pilot_20260313/hard/fixed`

目前 hard 主线 5 个 seed 的 `prune_only` phase 标签如下：

| `prune_only` phase | seeds |
| --- | --- |
| `prune0+2` | `0` |
| `prune0_only` | `1, 2, 3` |
| `no_prune` | `4` |

其中 follow-up 锚点刚好覆盖三类相：

- `seed_0 -> prune0+2`
- `seed_1 -> prune0_only`
- `seed_4 -> no_prune`

可复现的数据分组表由脚本 `scripts/build_unpruned_phase_alignment.py` 生成，输出到：

- `research/analysis/studies/unpruned_phase_differentiation_20260316/phase_seed_alignment.csv`

面向人工浏览的 phase 分组目录由脚本 `scripts/build_unpruned_phase_browser.py` 生成，输出到：

- `results/studies/unpruned_phase_differentiation_20260316/by_prune_phase/`

这个目录会直接按下面的结构重排：

- `prune0+2/seed_0/`
- `prune0_only/seed_1/`
- `prune0_only/seed_2/`
- `prune0_only/seed_3/`
- `no_prune/seed_4/`

每个 `seed_*` 目录里都会直接放好：

- `fixed_loss.png`、`fixed_sizes.png`、`fixed_active_neurons.png` 等 `fixed` 图表
- `prune_loss.png`、`prune_sizes.png`、`prune_active_neurons.png` 等 paired `prune_only` 图表
- `fixed_run/` 和 `prune_run/` 两个完整运行目录的软链接
- 一个简短的 `README.md`，概括这个 seed 的 paired 指标

面向 phase 级比较的对比图由脚本 `scripts/build_unpruned_phase_comparison_study.py` 生成，输出到：

- `results/studies/unpruned_phase_differentiation_20260316/primary/`
- `results/studies/unpruned_phase_differentiation_20260316/supplementary/`

这里会把当前 `fixed` 运行按 paired `prune_only` phase 分组后，统一画出：

- endpoint summary
- speed summary
- best-to-final degradation
- post-best drift rate
- degradation vs layer-2 share drift
- active drift
- share drift after best
- active share dynamics
- active dynamics
- size dynamics
- parameter-count dynamics
- turnover control
- train / test loss trajectories

面向“大量 seed 探相 + paired fixed 对照”的批量运行脚本是：

- `scripts/run_unpruned_phase_seed_sweep.py`

推荐分两步用：

1. 先用 `prune_only` 大范围探相，先看哪些 phase 实际出现、比例如何。
2. 再对这些 seed 补跑 `fixed`，生成 paired alignment、浏览目录和 phase 对比图。

典型命令：

```bash
python3 scripts/run_unpruned_phase_seed_sweep.py \
  --results-root results/followup_20260317/unpruned_phase_seed_sweep \
  --task hard \
  --modes prune_only \
  --seed-start 0 \
  --seed-end 199
```

如果决定直接把 `fixed` 也一起跑：

```bash
python3 scripts/run_unpruned_phase_seed_sweep.py \
  --results-root results/followup_20260317/unpruned_phase_seed_sweep \
  --task hard \
  --modes prune_only fixed \
  --seed-start 0 \
  --seed-end 199
```

这个脚本结束后会在：

- `results/followup_20260317/unpruned_phase_seed_sweep/hard/unpruned_phase_seed_sweep_analysis/`

自动写出：

- `prune_only_phase_inventory.csv`
- `prune_only_phase_counts.csv`
- `phase_seed_alignment.csv`（当 paired `fixed` 存在时）
- `by_prune_phase/`
- `phase_comparison/`

## 既有实验如何归入当前研究

当前这条“标准不剪枝网络的潜在相分化”研究线，不是从零开始，而是把之前已经做过的实验重新放进一个更清楚的因果框架里。建议按下面五类来理解：

| 既有实验 | 分类 | 在当前研究中的作用 |
| --- | --- | --- |
| `followup_hard_analysis_20260313` | 直接输入 | 这是当前研究最早的 seed 锚点来源。它先给出了 `prune_only` 在 `hard` 上的主三相样例，便于我们把同 seed 的 `fixed` 运行重排；而当前大 sweep 则把这个锚点库扩展成更完整的 phase 清单。 |
| `structural_phase_effects_20260314` | 直接前置 | 这是当前研究的 taxonomy 来源，提供了 `prune0+2 / prune0_only / no_prune` 这套相语言，也提供了“phase 早定、差异主要体现在 post-best 稳定性”的核心直觉。当前大 sweep 新浮现出的 `prune2_only` 可被视为对原 taxonomy 的扩展，而不是对其否定。 |
| `prune_only_vs_fixed_final_loss_20260314` | 直接支持 | 这个 study 负责说明 `prune_only` 和 `fixed` 的终点行为确实不同，因此“为什么不同”成为值得研究的问题。当前研究是在这个比较之上再往机制层推进。 |
| `phase_freeze_interventions_pilot_20260314` | 机制支持 | 它不直接研究 `fixed`，但它支持“late structural motion / post-best dynamics 可能是因果关键”这个机制判断。当前研究后续的 `shadow prune` 或冻结实验可以直接继承这条逻辑。 |
| `seed_pattern_review_20260313` | 历史前驱 | 这是较早的 pattern review，说明“seed 会把训练带入不同结构模式”这个现象不是偶然发现，但它现在应被视为历史前驱，而不是主证据。 |
| `prune_only_vs_grow_final_loss_20260314` | 边界条件 | 这个 study 用来回答当前现象是否只是 `prune_only vs fixed` 特有，还是更广泛地和 grow 系统的终点稳定性有关。它是边界条件，不是主干证据。 |
| `task_atlas_20260313` | 外部边界 | 它告诉我们当前现象是否只在 `hard` 上明显，还是能扩展到 `simple / medium`。因此它属于外部有效性和任务边界，而不是 phase 主证据。 |
| `mainline_core_summary_20260313` | 协议背景 | 它定义了主线 validation-selected 协议和主数据池。当前研究虽然是 phase 视角，但仍然要继承这套主协议，避免又回到不一致的 legacy 终点口径。 |
| `mainline_hard_diagnostics_20260313` | 诊断背景 | 这个包适合给当前研究提供 hard-task 的辅助图和诊断现象，但它不是围绕 phase 组织的，所以只能作为背景而不是主证据。 |
| `experiment_review_20260313` | 导航整合 | 它适合做项目级总览和回顾，但对当前研究不承担独立证据功能，更像导航页。 |

压缩成一句话就是：

- `followup_hard_analysis_20260313` + `structural_phase_effects_20260314` + `prune_only_vs_fixed_final_loss_20260314` 构成当前研究的主干；
- `phase_freeze_interventions_pilot_20260314` 和 `seed_pattern_review_20260313` 提供机制与历史支撑；
- 其余包主要负责边界、协议和导航。

## 目前对 `fixed` 的方向性观察

下面这些数值只是“研究起点”，不是统计结论，因为当前 `prune0+2` 和 `no_prune` 组都只有 `n=1`。

| 由同 seed `prune_only` 赋予的标签 | seeds | `fixed` median selected test | `fixed` median final test | `fixed` median final-selected | `fixed` median best epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| `prune0+2` | `0` | `1.43e-4` | `2.14e-4` | `7.10e-5` | `2991` |
| `prune0_only` | `1,2,3` | `1.16e-4` | `3.21e-4` | `2.09e-4` | `2858` |
| `no_prune` | `4` | `1.46e-4` | `6.00e-4` | `4.53e-4` | `2442` |

这组数的一个重要信号是：

- `fixed` 的 selected/best 表现并没有像 final 表现那样明显分开。
- 差异更像是出现在 `best -> final` 的退化幅度，以及 late-stage 的稳定性上。
- 这与我们在结构相研究里得到的“相之间主要区别不在最优点，而在 post-best 稳定性”是同方向的。

如果后续加大 seed 数后这个模式仍然成立，那么“潜在相标签先决定后期命运，再由剪枝把它放大”就会成为一个很强的解释。

## 文献依据

下面这些文献直接支撑这个研究框架，而不是泛泛地“和神经网络有关”：

1. Li et al., 2016, *Convergent Learning: Do different neural networks learn the same representations?*  
   链接: <https://arxiv.org/abs/1511.07543>  
   启发：不同初始化的网络会学到部分相似、部分不对齐的表示，因此跨 seed 比较不能只盯住单个神经元身份，更适合用 layer-level 统计和表示相似度。

2. Kornblith et al., 2019, *Similarity of Neural Network Representations Revisited*  
   链接: <https://proceedings.mlr.press/v97/kornblith19a.html>  
   启发：`CKA` 是比较不同网络内部表示的稳健工具。后续如果我们想验证不同 phase 标签下的 `fixed` 是否真的进入不同表征区，`CKA` 应该作为主方法之一。

3. Frankle and Carbin, 2019, *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*  
   链接: <https://openreview.net/forum?id=rJl-b3RcF7>  
   启发：dense 初始化里本来就嵌着可训练的稀疏子网络。我们把 `prune_only` 的 phase 当作 dense 网络潜在子网络结构的探针，是有理论和经验依据的。

4. Draxler et al., 2018, *Essentially No Barriers in Neural Network Energy Landscape*  
   链接: <https://proceedings.mlr.press/v80/draxler18a.html>  
   启发：不同 seed 的解在 loss landscape 中可能是连通的，所以“最终能连通”并不等于“训练路径和内部功能组织相同”。这正好支持我们把关注点放在训练路径和 late drift 上。

5. Frankle et al., 2020, *Linear Mode Connectivity and the Lottery Ticket Hypothesis*  
   链接: <https://proceedings.mlr.press/v119/frankle20a.html>  
   启发：早期训练阶段对后续轨迹非常关键，SGD 噪声的影响在训练早段更大。我们的“phase early commit”直觉与这个结论是相容的。

6. You et al., 2020, *Drawing Early-Bird Tickets: Toward More Efficient Training of Deep Networks*  
   链接: <https://arxiv.org/abs/1909.11957>  
   启发：稀疏结构的可识别信号可以在训练很早阶段稳定下来。这支持我们在 `fixed` 中寻找早期的潜在相标记，而不是只看最终终点。

7. Wang et al., 2025, *The Butterfly Effect of Optimizing Neural Networks*  
   链接: <https://openreview.net/forum?id=1jmd-B3Yg4>  
   启发：极小扰动都可能把训练推向不同轨道，因此 seed 不是“噪声变量”，而是我们这里要主动利用的因果入口。

8. Jinnai et al., 2025, *Break-even Points of DNN Training*  
   链接: <https://openreview.net/forum?id=fvSlRma0qX>  
   启发：深度网络训练过程中可能存在可辨识的阶段切换点。我们后续可以把 `fixed` 的“commit point”定义成某些 drift 或 shadow-prune 指标首次稳定的时刻，而不必拘泥于真实剪枝事件。

## 核心假设

### H1：潜在相假设

如果一个 seed 在 `prune_only` 中落入 `prune0+2`，那么同 seed 的 `fixed` 运行也更可能表现出：

- 更强的 late-stage 功能重分配；
- 与后层相关的更高 active-share 或表示偏移；
- 更明显的 final stability 问题。

### H2：放大而非创造假设

`prune_only` 更可能是在放大 dense 网络原本就存在的分化，而不是从零创造相位。

支持证据将表现为：

- `fixed` 按 paired `prune_only` phase 分组后，selected 指标差异小，但 final 指标和 drift 指标差异大；
- 分组差异在早期或中期就能看到先兆，而不是只在剪枝后才突然出现。

### H3：表示层面可分假设

即使不同组的最终损失差别不大，它们的内部表示几何也会不同。也就是说，`functionally similar` 不等于 `representationally identical`。

## 实验设计

### 第一层：已有结果的观测性重排

不改代码，先做现成数据的 phase-conditioned analysis。

主读出：

- `final_test_loss`
- `test_loss_at_best_val` 或 metrics 中的 best-val 对应 test
- `final - selected`
- `final test - final train gap`
- `best_epoch`
- `active_*` 的 best/final/drift
- 各层 active-share 的 best/final/drift

分组方式：

- 全量池：按 `phase_seed_alignment.csv` 中的 `prune_only_phase` 分组
- 锚点池：只取 `seed_0 / seed_1 / seed_4`，每类相各一个代表

### 第二层：在 `fixed` 中加入 shadow prune 记录

这是这个分支最关键的下一步代码工作。

做法：

- 在 `fixed` 模式下，照常计算 `importance -> EMA -> candidate -> ablation`
- 但不真正删除神经元
- 只记录“本轮本来会被 `prune_only` 剪掉哪些神经元”

新增建议指标：

- `shadow_candidate_i`
- `shadow_prunable_i`
- `shadow_total_prunable`
- `shadow_ablation_delta_*`
- `shadow_commit_epoch`

这样就能回答：

- 哪些 `fixed` 运行其实已经在内部走到了“可剪枝相”？
- 这些 would-be-pruned 神经元后面是继续沉没，还是会被重新利用？

### 第三层：因果 follow-up

当 shadow-prune 记录完成后，再做更强的 follow-up：

- 对 `fixed` 运行在 `shadow_commit_epoch` 之后冻结某些层或冻结学习率
- 比较是否能减少 late degradation
- 对 `prune_only` 和 `fixed` 做 seed-matched 的表示相似度轨迹比较

## 判别逻辑

我们希望用下面的逻辑来判别“揭示”还是“创造”：

- 如果 `fixed` 在 paired phase 标签下已经可分，那么剪枝更像揭示/放大。
- 如果 `fixed` 不可分，而 `prune_only` 明显可分，那么剪枝本身更可能是主要致因。
- 如果 `fixed` 只在表示层面可分、但在终点 loss 上不太可分，那么结论应写成“潜在表征分化先于明显性能分化”。
- 如果 `fixed` 的分组差异主要体现在 `best -> final`，而不是 `selected`，那么我们就可以把研究重点继续放在 late stability，而不是 peak performance。

## 当前限制

- 当前 hard 主线只有 5 个 seed，且 phase 分布不平衡：`1 / 3 / 1`。
- 现在的 phase 标签来自 paired `prune_only`，还不是 `fixed` 原生 taxonomy。
- 现有 `fixed` logs 还没有 `shadow prune`、`CKA`、`SVCCA` 之类的内部表示证据。
- 目前的方向性观察只能支持“值得研究”，还不能支持统计显著的论文表述。

## 近期执行顺序

1. 用现有 `phase_seed_alignment.csv` 先画出 `fixed` 的 phase-conditioned endpoint 和 drift 图。
2. 给 `fixed` 加 `shadow prune` 记录，尽量不改变现有训练协议。
3. 扩 seed，使每个 phase 至少有 5 个 paired runs，避免现在的 `1 / 3 / 1` 失衡。
4. 在扩 seed 后再加入 `CKA` / `SVCCA`，判断 phase 是否已经对应不同表征轨道。

## 本 study 的一句话定位

这不是“再比较一次 `fixed` 和 `prune_only` 谁更好”，而是把 `prune_only` 当成 dense 网络潜在训练路径的显影剂，反过来研究标准不剪枝网络内部是否已经带有可被剪枝系统放大的相分化。
