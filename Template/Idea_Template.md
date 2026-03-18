# 💡 Research Idea: [项目简称/名称]

> **Agent Instruction:** 这是项目的核心灵魂。在执行任何代码修改或实验分析前，请务必完整阅读此文件，确保所有动作符合本研究的逻辑框架。

---

## 1. 核心概述 (The Essence)

- **一句话总结 (One-Sentence Summary):**
  > [例如：通过 [方法名] 解决 [具体问题]，在 [具体场景] 下提升 [核心指标] 性能。]

- **一段话摘要 (One-Paragraph Abstract):**
  [在这里简述研究的背景、核心瓶颈、你的创新点以及最终预期的科学贡献。确保 Agent 能理解研究的“Vibe”和核心逻辑。]

---

## 2. 深度阐述 (Deep Dive)

### 2.1 背景与动机 (Background & Motivation)
- **大背景:** [如：AIGC 时代下的内容安全或智能体防御现状。]
- **核心痛点:** [目前的技术（如 NCFM）在什么情况下会失效？为什么现有的防御手段不够？]
- **动力来源:** [为什么现在解决这个问题至关重要？]

### 2.2 相关工作现状 (State of the Art - SOTA)
- **主流方法 A:** [简介及其局限性]
- **主流方法 B:** [简介及其局限性]
- **Gap:** [目前的文献中还缺少什么？]

### 2.3 研究目标 (Research Objectives)
1. [目标 1: 理论证明...]
2. [目标 2: 开发一个鲁棒的 Agent 框架...]
3. [目标 3: 在标准数据集上超越 Baseline %...]

---

## 3. 技术实现 (Methodology)

### 3.1 理论基础 (Theoretical Foundation)
[使用 LaTeX 描述核心公式]
$$\mathcal{L}_{total} = \alpha \mathcal{L}_{task} + \beta \mathcal{L}_{defense}$$
- **关键假设:** [如：假设对抗噪声是有限边界的。]

### 3.2 核心方法 (Proposed Method: [名称])
- **架构设计:** [Agent 应该关注的核心模块，如 Encoder-Decoder 或 RL Policy。]
- **创新算法步骤:** 1. ...
  2. ...

### 3.3 基准对比 (Baselines)
- **Baseline 1:** [名称] (来源: [论文/链接])
- **Baseline 2:** [名称] (如：Vanilla NCFM)
- **比较维度:** [准确率, 推理延迟, 鲁棒性得分等]

---

## 4. 实验设计 (Experimental Design)

### 4.1 数据集 (Datasets)
- [ ] [数据集 A] (用途: 训练/验证)
- [ ] [数据集 B] (用途: 跨域测试)

### 4.2 实验流程 (Execution Pipeline)
1. **Pilot Study:** [初步验证核心假设的小规模实验。]
2. **Main Experiment:** [全量数据训练与对比。]
3. **Ablation Study:** [消融实验设计，验证每个组件的有效性。]

### 4.3 评价指标 (Metrics)
- **Primary:** [主要指标]
- **Secondary:** [次要指标]

---

## 5. 预期效果与影响 (Expected Outcomes)
- **预期结果:** [例如：在对抗攻击下保持 80% 以上的成功率。]
- **潜在贡献:** [对社区或后续研究的影响。]

---

## 6. Agent 追踪专区 (Agent Traceability)
> **注意:** 本节由 Agent 实时更新，记录 Idea 的演进。

- **[2026-XX-XX] 演进记录:** Agent 修改了核心公式中的正则项，理由是初步实验发现收敛速度过慢。
- **当前瓶颈:** [Agent 自动填写：目前实验 A 遇到的主要阻碍。]