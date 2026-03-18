# 🧠 Method: MemSifter 偏好错误分解诊断框架

> **Agent Instruction:** 本文件当前记录的不是最终的 `Preference Arbiter` 算法，而是“在是否值得做 arbiter 之前，我们如何系统诊断 MemSifter 错误来源”的方法说明。

---

## 1. 方法总览（High-Level Overview）

- **一句话方法总结：**  
  我们先用一个分阶段、可量化的诊断框架，把 `PersonaMem` / `PersonaMem-v2` 上的错误拆成 embedding 未召回、rerank 丢失、以及 post-rerank 回答策略失效三类，再据此判断是否值得在 MemSifter 后面增加 `Preference Arbiter`。
- **核心直觉（Intuition）：**  
  如果相关 session 在 embedding 和 rerank 后都已经进入最终上下文，但模型仍然答错、泄漏 `ask_to_forget` 元信息，说明问题已经不主要是 retrieval，而更像是 post-rerank 的使用策略。反过来，如果 gold session 在 embedding 或 rerank 阶段就频繁消失，那么直接做一个下游 arbiter 很可能治标不治本。
- **数学层级定位（Optional）：**  
  - ⚖️ **解释层级（中等）：** 当前框架以形式化错误分解和可检验命题为主，目的是约束研究方向，而不是对尚未实现的方法硬写理论。

---

## 2. 记号与问题设定（Notation & Problem Setup）

### 2.1 记号约定（Notation）

- 对于第 `i` 个 benchmark 样本，记为  
  \[
  x_i = (q_i, H_i, y_i, m_i)
  \]
  其中：
  - `q_i`：用户问题；
  - `H_i = (s_{i1}, s_{i2}, \dots, s_{in_i})`：该样本对应的历史 session 序列；
  - `y_i`：标准答案；
  - `m_i`：元数据，如 `updated`、`pref_type`、`who` 等。
- 若后续需要显式时间信号，记 `t_{ij}` 为 session `s_{ij}` 的相对时间顺序。当前 `PersonaMem` / `PersonaMem-v2` 数据中暂无稳定可直接使用的显式时间戳时，默认先用 `H_i` 中的 session 顺序作为弱时间代理。
- 若数据集提供可对齐的 gold session 信息，记 gold session 集合为  
  \[
  G_i \subseteq \{1,2,\dots,n_i\}
  \]
- embedding 预筛选后的候选集合记为  
  \[
  C_i^{\text{emb}} \subseteq \{1,2,\dots,n_i\}, \quad |C_i^{\text{emb}}| = K_e
  \]
  当前实验中 `K_e = 20`。
- rerank 后保留下来的候选集合记为  
  \[
  C_i^{\text{rank}} \subseteq C_i^{\text{emb}}, \quad |C_i^{\text{rank}}| = K_r
  \]
  当前实验中 `K_r = 5`。
- 最终 chat 阶段生成的答案记为  
  \[
  \hat{a}_i
  \]

### 2.2 问题形式化（Problem Formalization）

- **任务目标：**  
  当前阶段不是直接优化新模型，而是要回答一个更基础的问题：
  \[
  \text{MemSifter 在目标 benchmark 上的主要失败，究竟发生在哪一层？}
  \]
  具体拆成三类候选原因：
  \[
  \text{Embedding Miss},\quad \text{Rerank Drop},\quad \text{Post-Rerank Misuse}
  \]

- **阶段命中指标：**
  \[
  h_i^{\text{emb}} = \mathbb{1}[G_i \cap C_i^{\text{emb}} \neq \varnothing]
  \]
  \[
  h_i^{\text{rank}} = \mathbb{1}[G_i \cap C_i^{\text{rank}} \neq \varnothing]
  \]
  其中：
  - `h_i^{emb} = 1` 表示 embedding 至少召回了一个 gold session；
  - `h_i^{rank} = 1` 表示 rerank 之后 gold session 仍然被保留。

- **答案正确性指标：**
  \[
  c_i = \mathbb{1}[\hat{a}_i \text{ judged correct}]
  \]
  当前项目里，`PersonaMem-v2` 已经可以做阶段命中诊断，但 `PersonaMem` 的官方答案仍是多选标签，因此 `c_i` 在两个数据集上的统一自动计算尚未完全落地。

- **错误标签定义：**
  \[
  e_i =
  \begin{cases}
  \text{EmbeddingMiss}, & h_i^{\text{emb}} = 0 \\
  \text{RerankDrop}, & h_i^{\text{emb}} = 1,\ h_i^{\text{rank}} = 0 \\
  \text{PostRerankMisuse}, & h_i^{\text{rank}} = 1,\ c_i = 0 \\
  \text{EmptyAnswer}, & \hat{a}_i = \varnothing \\
  \text{ForgetLeak}, & m_i.\text{pref\_type}=\texttt{ask\_to\_forget} \land \hat{a}_i \text{ 泄漏元信息}
  \end{cases}
  \]
  其中 `EmptyAnswer` 和 `ForgetLeak` 当前可以视为 `PostRerankMisuse` 的特例或细分标签。

- **当前核心研究判据：**  
  设“冲突偏好相关错误”占所有失败样本的比例为
  \[
  \rho_{\text{conflict}} = \frac{\#\{\text{conflict-related failures}\}}{\#\{\text{all failures}\}}
  \]
  若
  \[
  \rho_{\text{conflict}} \ge 0.3
  \]
  则支持继续推进 `Preference Arbiter` 作为主线；否则需要重新考虑是否优先修复 rerank 或 chat policy。

---

## 3. 诊断流程（Diagnostic Procedure）

1. 在 benchmark 上运行标准 pipeline：  
   \[
   q_i \rightarrow \text{Embedding Top-}K_e \rightarrow \text{Rerank Top-}K_r \rightarrow \text{Chat}
   \]
2. 对每个样本记录：
   - `embedding_candidates`
   - `reranked_sessions`
   - `predicted_answer`
   - `gold_session_indices`（若可用）
   - `metadata`
3. 计算总体与子集统计：
   - 总体 `embedding_gold_hit_rate`
   - 总体 `rerank_gold_hit_rate`
   - 按 `updated`、`pref_type`、`who` 切分的阶段命中率
4. 对异常回答做人工标注：
   - `ask_to_forget` 泄漏
   - 空回答
   - `who/self` 归属错误
   - polarity 错误
   - rerank 正确但回答未正确使用
5. 根据分解结果决定下一步方法方向：
   - 若 `EmbeddingMiss` 为主，优先改召回；
   - 若 `RerankDrop` 为主，优先改 reranker；
   - 若 `PostRerankMisuse` 为主，才更有理由推进 arbiter 或回答策略模块。

---

## 4. 可检验命题（Testable Propositions）

### 4.1 Proposition 1：高命中子集上的剩余错误不能主要靠 retrieval 修复

- **命题：**  
  对某个子集 `A`，若有
  \[
  \forall i \in A,\ h_i^{\text{emb}} = 1,\ h_i^{\text{rank}} = 1
  \]
  或至少这两个命中率已经接近 `1`，则该子集中的剩余错误不应再主要归因于 retrieval，本质上需要 post-rerank 的解释或修复。

- **证明思路（Proof Sketch）：**
  1. `h_i^{rank} = 1` 表示 gold session 已经进入最终供 chat 使用的上下文。
  2. 因此 embedding 和 rerank 已经完成“把相关信息送到下游”的职责。
  3. 若样本仍答错、空回答或泄漏不自然元信息，则 retrieval-only 改进无法解释该剩余误差。
  4. 所以这部分问题应优先在 post-rerank 模块中寻找原因，例如回答策略、冲突消解、格式约束或输出解析。

### 4.2 Corollary 1：当前 `updated=True / ask_to_forget` 子集更像 post-rerank 问题

- 在当前 `PersonaMem-v2` 前 100 条样本中：
  \[
  \#\{i: updated_i=True\} = 20
  \]
  且这 `20` 条全部属于 `pref\_type = ask\_to\_forget`。
- 这一子集上观察到：
  \[
  \text{Embedding hit} = 20/20 = 1.0
  \]
  \[
  \text{Rerank hit} = 20/20 = 1.0
  \]
- 因此，当前这类样本若存在不自然回答、元信息泄漏或空输出，retrieval-only 修复不太可能是主要解法。

### 4.3 Corollary 2：当前 `updated=False` 子集更像 rerank 瓶颈

- 在当前 `PersonaMem-v2` 前 100 条中，`updated=False` 且有 gold 对齐的样本共有 `72` 条。
- 这一子集上：
  \[
  \text{Embedding hit} = 53/72 \approx 0.7361
  \]
  \[
  \text{Rerank hit} = 35/72 \approx 0.4861
  \]
- 说明在这一类非更新型隐式偏好样本上，gold session 更容易在 rerank 后丢失，因此 rerank 是更直接的瓶颈候选。

---

## 5. 理论解释：为什么这个诊断框架有效？（Why It Works）

- **机制解释：**  
  该框架把“模型低分”拆成多阶段可观测事件，而不是直接把所有失败都归因给一个抽象故事。这样可以避免在主假设尚未证实时就提前锁定方法方向。
- **对比分析：**  
  与“看最终 F1 然后直接设计新模块”的做法相比，这个框架能把问题区分成：
  - 召回不足；
  - rerank 误删；
  - 回答阶段误用；
  这三类对应完全不同的修复路径。
- **信息流视角：**  
  `Embedding -> Rerank -> Chat` 可以看成一条信息传输链。若 gold 信息在前段已经丢失，后段无法补救；若 gold 信息已保留到末端而回答仍异常，则错误一定来自下游使用方式。

---

## 6. 与实验设计的对齐（Link to Experiments）

- **可验证的理论结论：**
  - `h_i^{emb}` ↔ 实验中的 `embedding_gold_hit_rate`
  - `h_i^{rank}` ↔ 实验中的 `rerank_gold_hit_rate`
  - `EmptyAnswer` ↔ `predicted_answer == ""`
  - `ForgetLeak` ↔ 对 `ask_to_forget` 子集的人工回答检查
- **实验指标与数学量的对应表：**
  - `K_e = 20` ↔ 当前 embedding 预筛选候选数
  - `K_r = 5` ↔ 当前 rerank 保留候选数
  - `c_i` ↔ 后续将接入的答案正确性评分
  - `\rho_{\text{conflict}}` ↔ 后续失败标签统计中的“冲突偏好类错误占比”
- **当前已观测到的事实：**
  - `PersonaMem-v2` 前 100 条样本中，gold 可对齐样本为 `92` 条。
  - 总体 embedding 命中为 `73 / 92 ≈ 79.35%`。
  - 总体 rerank 命中为 `55 / 92 ≈ 59.78%`。
  - `updated=True / ask_to_forget` 子集为 `20` 条，embedding 和 rerank 命中均为 `100%`。
  - `updated=False` 子集为 `72` 条，embedding 为 `53 / 72 ≈ 73.61%`，rerank 为 `35 / 72 ≈ 48.61%`。
  - 当前 `PersonaMem-v2` 100 条里出现 `3` 条空回答。
  - `PersonaMem` 当前尚未形成与官方口径对齐的 `c_i` 计算方式，因为 benchmark 是多选标签而当前脚本输出自由文本。

---

## 7. Agent 追踪备注（Agent Notes）

- **[2026-03-18] 数学结构更新：**  
  新建本文件，并将当前方法定义为“阶段化错误分解诊断框架”，而不是尚未实现的 `Preference Arbiter` 正式算法。
- **[2026-03-18] 当前数学瓶颈：**  
  - `PersonaMem` 的答案正确性 `c_i` 还未和官方口径对齐；
  - “冲突偏好相关错误”的判定规则还未正式落盘成统一标签体系；
  - 当前样本覆盖 persona 数量较少，尚不足以支撑总体结论。
- **[2026-03-18] 时间信号约束：**
  - 当前 `PersonaMem` / `PersonaMem-v2` 数据未提供稳定的显式时间戳字段；
  - 若继续推进“时序仲裁”，默认先以 session 顺序作为弱时间代理，再评估是否需要额外构造时间特征。
- **[2026-03-18] 下一次方法更新触发条件：**  
  只有在完成更大样本的失败标签统计后，才适合把方法从“诊断框架”升级为“arbiter 模块设计”，并正式补充其输入输出、目标函数与 ablation 对齐方式。
