# 全流程执行追踪 + L4（SQL 选择）对齐 SIRIUS

本文记录本次提交做过的两类工作：一是新增「全流程详尽执行追踪」子系统，二是把 SQL 选择阶段（L4）按 SIRIUS 论文 §3.5 重新对齐（P0 + P1）。

---

## 一、全流程执行追踪（tracing）

目标：把 few_shot → value_retrieval → schema_linking → sql_generation → sql_revision → sql_selection → evaluation 七个阶段里每一个可分析细节都落盘，便于事后复盘。埋点是无条件写的，关闭追踪时自动退化为 no-op。

### 1.1 落盘位置与格式

- 目录：`Path(stage_config.save_path).parent / "trace"`，同一 run 的所有阶段共用一个目录。
- 文件：`events.jsonl`（每行一个 JSON 事件）+ `meta.json`（run 元信息）。
- 事件结构：

  ```json
  {"seq": 12, "ts": "2026-10-30T21:00:00.123+08:00", "elapsed_ms": 1234.5,
   "run_id": "20261030T205900", "stage": "sql_generation", "item_id": "42",
   "event": "llm_response", "data": {...}}
  ```

- 开关：`DEEPEYE_TRACE=0` 关闭；`DEEPEYE_TRACE_DIR` 覆盖落盘目录。
- 写入方式：`TraceWriter` 使用常驻 daemon 线程 `deepeye-trace-writer` + `queue.Queue`，逐行 flush，`atexit` 时 close，主流程不被 IO 阻塞。

### 1.2 关键实现点

| 问题 | 处理 |
| --- | --- |
| 多阶段各自 `configure_tracing` | 指向同一目录时复用已有 writer，不重复打开 |
| 同一文件被多段进程追加 | `TraceWriter.__init__` 从文件末尾最大 `seq` 续号，避免重复编号 |
| 线程池丢失上下文 | `submit_with_trace(executor, fn, ...)` 用 `contextvars.copy_context()` 后提交 `ctx.run(fn, ...)` |
| `ProcessPoolExecutor` 无法透传 contextvars | evaluation 改由子进程返回明细 dict，父进程统一写 trace |
| 超大对象（prompt、结果表） | `_jsonable` 递归序列化，`_MAX_STR=40000` / `_MAX_LIST=500` / `_MAX_DEPTH=8`，超限截断并标注 |

代码：`app/tracing/tracer.py`、`app/tracing/executor.py`、`app/tracing/__init__.py`。

### 1.3 各阶段埋点

- **LLM 调用**（`app/llm/llm.py`）：`llm_request`（完整 messages/prompt、采样参数）、`llm_response`（原始返回、token 用量、耗时）、`llm_error`（异常原文）。
- **解析重试**（`app/llm_extractor/extractor.py`）：每次 `extract_with_retry` 的解析成功/失败、重试序号、失败原文，`trace_label` 标明用途（如 `sql_selection.pairwise`）。
- **SQL 执行**（`app/services/execution_service.py`）：`sql_execute`（SQL、result_type、错误、行数、耗时）、`measure_time`。
- **few_shot**（`masker.py` / `runtime.py` / `preliminary_sql.py`）：mask 前后的问题、检索到的示例、preliminary SQL 的选择过程。
- **value_retrieval**：抽取的关键词、候选值、命中情况、修改后的 schema。
- **schema_linking**：direct / reversed / value 三类链接器的输入与产出、合并后的最终链接结果、各口径 recall。
- **sql_generation**：每条候选的生成来源（dc / skeleton / icl）、发送的 prompt、模型返回、清洗（stripping level）后的候选。
- **sql_revision**：每条候选进入时的 SQL、各 checker / typed repair 的分类（clean / runtime / timeout / empty）、修复前后 SQL、重执行结果、pool recovery、最终候选清单及来源标签。
- **sql_selection**：执行摘要、结果一致性（含 `max_class_size`）、top-k、pairwise 计划与每次比较（含 A/B 换位）、胜率矩阵、合并分数、置信门控、Stage 2 结构覆盖、最终选择。
- **evaluation**（`runner/evaluation.py`）：`evaluation_start` / `evaluation_item` / `evaluation_item_error` / `evaluation_end`。逐题记录预测与金标 SQL 的执行摘要、是否正确、判定原因、缺失/多余行（`_EVAL_ROW_PREVIEW=20`），结束时汇总 accuracy、mismatch_ids、gold_failed_ids、crashed_ids。

---

## 二、L4（SQL 选择）对齐 SIRIUS

对照论文 §3.5 后，原实现有 7 处偏离。本次做 P0 + P1 三项，其余留作后续。

### 2.1 P0：门控语义（第 ④ 处）

- 旧：`consistency gap >= 0.3` 就直接选 top-1，pairwise **不跑**（`gap_shortcut` 分支）。
- 新：**候选数 >1 时 pairwise 一律必跑**；门控只决定是否上 Stage 2：
  - `merged` 与 `S_rm` 的 top-2 gap **都恰好为 0**（`tie_tol = 1e-9`）；
  - `1 < K ≤ 6`；
  - 最大结果等价类 `≤ 8`（`Tmax`）。
- 事件：`selection_gate`（含各项条件与实测 gap）。`style=gap_shortcut` 事件彻底移除。

### 2.2 P1①：融合方式（第 ① 处）

- 旧：`ranking = mean(win) * (consistency / Σconsistency)`（**相乘**）。
- 新：`S_coarse = S_maj + S_rm`（**相加**，论文 Eq.4 / Eq.23）。
- `selection_ranking` 事件改为输出 `mean_win_scores` / `consistency_scores` / `merged_scores` / `ranking` / `sources`。

### 2.3 P1⑦：Stage 2 结构支持度（第 ⑦ 处）

- 旧：同结构签名计数 + `0.1 * (count-1)/(n-1)` bonus，`epsilon=0.02` 近似并列即重排。
- 新：`ast_signature.py` 新增
  - `compute_cross_source_support(sqls, sources)`：按结构签名分组，统计组内**不同来源标签**数，即论文的 `srcgrp`；
  - `structural_support_select(sqls, merged_scores, sources, default_idx, tie_tol)`：只在合并分数**完全并列**的候选间比较，来源支持**严格更强**才覆盖，否则保留 top-1。
- 事件：`selection_structural_override`（含 `tied_indices` / `supports` / `signatures` / `overridden`）。

### 2.4 P1③：Elo 迭代求 `S_rm`（第 ③ 处）

- 旧：对每对候选的多次投票取均值得到 `robust_win_matrix`，再按行均值排序。单轮 Bradley-Terry 简化，**没有胜负传递性**（A 胜 B、B 胜 C 不会推高 A 对 C）。
- 新：`_compute_elo_reward_scores(robust_win_matrix)` 对 `robust_win_matrix`（对称零和）做 Elo 迭代（`k=0.5`、最多 200 轮、`scale=400`，以 `1e-6` 提前收敛），收敛后取每个候选**对全场的期望得分**作为 `S_rm`，与旧的胜率同处 `[0,1]` 量纲，可直接与 `S_maj` 相加。
- `selection_ranking` 事件新增 `rm_scores` / `elo_ratings`，同时保留 `mean_win_scores` 便于对比分析。
- 说明：`filter_top_k_sql=2` 时只有一对候选，Elo 与行均值排序等价，结论不变；该改动主要影响 `K≥3` 的题目。

### 2.5 配套：候选来源（provenance）链路

Stage 2 需要知道每条候选来自哪条生成路线，因此把来源标签一路带到选择阶段：

- `app/dataset/dataset.py`：`DataItem.sql_candidate_sources`（与 `sql_candidates_after_revision` 对齐）。
- `app/dataset/artifacts.py`：`SQLGenerationArtifact` / `SQLRevisionArtifact` 增加同名字段，并加入 `STAGE_ARTIFACT_FIELDS`（随 checkpoint/snapshot 持久化）。
- `sql_generation.py`：按 `dc / skeleton / icl` 顺序写入。
- `sql_revision.py`：按原顺序回填（保持重复与顺序），pool recovery 追加的候选标 `recovery`；旧快照缺失该字段时统一兜底 `unknown`。
- `sql_selection.py`：按 `sql → source` 映射对齐 top-k 候选。

### 2.6 P2②：裁判模型可配置（第 ② 处，部分）

- 旧：pairwise 直接用生成阶段的同一个 `self._llm`，属 self-judge。
- 新：`SQLSelectionConfig` 增加可选 `judge_llm`；配置层支持
  - `[sql_selection] judge_llm_profile = "<profile>"`（复用 `[llm_profiles.*]`），或
  - `[sql_selection.judge_llm]` 内联完整模型配置（model / base_url / api_key 等）。
- 裁判配置**独立解析**，不与生成模型合并（否则会继承 `extra_body = {enable_thinking = ...}` 这类 Qwen 专有字段，跨厂商模型会因此报错或行为异常）。
- Runner：`self._judge_llm = LLM(judge_llm_config)`，未配置时回退到 `self._llm`（行为与旧版一致）；`_compare_sqls` 的 prompt 构造、模型调用与 `fix_end_token` 全部走 `_judge_llm`。前置阶段（few-shot / 生成 / 修复）的模型完全不受影响。
- 换成 DeepSeek 的写法（已在 `config/local/Qwen3.6-27B/config-bird-dev.toml` 中作为注释示例）：

```toml
# 模型名以官方文档为准：deepseek-v4-flash / deepseek-v4-pro
# （deepseek-chat / deepseek-reasoner 已于 2026-07-24 弃用）
[sql_selection.judge_llm]
model = "deepseek-v4-flash"
base_url = "https://api.deepseek.com"
api_key = "sk-xxxx"
max_tokens = 8192
temperature = 0.2
n_call_strategy = "split"   # 不支持单请求 n>1，按 split 多次请求凑 5 票
extra_body = { thinking = { type = "disabled" } }   # 思考模式默认开启且忽略 temperature
```
- Trace：`selection_compare_prompt` 增加 `judge_model` / `judge_is_stage_llm`，便于事后确认该题是否由同一模型判定。
- 局限：当前环境只有一个真实可用的模型端点（dashscope `qwen3.6-27b`），把 judge 指到 `qwen36_instruct` 只是换了采样配置（thinking 关、temperature 0.7），**并非真正独立的模型**；论文那种 Qwen2.5-Coder-32B 微调 pairwise RM 需 GPU 训练，本机无法复现。

### 2.7 pairwise 裁判：改为「独立双标签判定」（论文 E.4 / Eq.25 / Eq.26）

从 trace 的 `llm_call_start.messages`（`trace_extra.label == "sql_selection.pairwise"`）里 dump 出实际发给裁判的 6743 字符 prompt，确认旧模板存在两处与论文相悖的设计，已按论文改掉：

| 旧设计 | 问题 | 论文做法（已采用） |
| --- | --- | --- |
| 文案写死「A 是 Top-1，置信度更高；判断不出来就选 A」 | `_compare_sqls` 在 `swapped=True` 时只交换 SQL 位置、文案不变，被展示为 A 的其实是 Top-2（甚至是被修复/被覆盖的那条），指令自相矛盾；更关键的是这种先验写进 prompt 后，位置交换根本无法抵消 | 论文 RM 输入只含 question + evidence + 相关表 schema + 每方前 k 行结果，**不含任何候选先验**；位置去偏靠对称地跑两种顺序（Eq.25） |
| 只允许输出 `A` / `B` | 解析器与 win matrix 本就支持 `TIE`，5 票采样被退化成强制二选一，平票信息丢失；论文 Table 9 中 correct,correct 占 9.2%、incorrect,incorrect 占 20.3%，双对/双错是合法判定 | 裁判对两条 SQL **各自独立**给出 `Correct` / `Incorrect`，投票由两个标签推导（Eq.26 的 wins-minus-losses 以标签差为基础） |

改动：

- `app/prompt/prompt_template.py`：`BR_PAIR_SELECTION_PROMPT` 整段重写。新增 `# Judging Rules`（各自按自身对错标注、两条独立判定、**两条都正确或都错误都是允许的**、展示顺序不含任何质量信息、明确 Incorrect 的四种情形）；输出改为
  ```xml
  <result>
  <sql1_judge>Correct or Incorrect</sql1_judge>
  <sql2_judge>Correct or Incorrect</sql2_judge>
  </result>
  ```
  输入段标签改为 `SQL Query 1 (sql1)` / `## Execution Result of sql1` / `SQL Query 2 (sql2)` / `## Execution Result of sql2`。占位符名（`QUERY_A/RESULT_A/QUERY_B/RESULT_B`）保持不变，`factory.format_br_pair_selection_prompt` 无需改动。
- `app/prompt/spider2_prompt_template.py`：`SPIDER2_BR_PAIR_SELECTION_PROMPT` 同步同样改造（该模板按 `db_type` 被 factory 选中，不同步会导致 BigQuery/Snowflake/SQLite 仍走旧逻辑）。
- `app/pipeline/sql_selection/sql_selection.py`：新增 `_normalize_judge_label`（`INCORRECT` 含 `CORRECT` 子串，必须先判），`_parse_llm_response` 改为解析两标签并推导投票：
  - `(C, I) → A`、`(I, C) → B`
  - `(C, C) → TIE`、`(I, I) → TIE`
  - 两标签缺失时回退兼容旧式单字母输出，再失败返回 `None`（该票作废，与论文 `Jp ∈ {1,0,⊥}` 的 ⊥ 语义一致）。
- `_update_win_matrix` 的 A/B/TIE 语义不变（`TIE` 双方各 0.5 分，`swapped` 时 A/B 互换），与新解析器天然兼容；仅新增 `vote_offset` 用于双顺序分槽位（见下）。

**位置去偏改为两种顺序都跑（Eq.25）**

旧实现每对候选掷一次硬币（`random.random() < 0.5`）只跑一种顺序，指望跨 pair 平均掉位置偏置。问题在粒度：随机发生在 **pair 粒度**而不是 vote 粒度，`_compare_sqls` 的 5 票共用同一个 `swapped`，所以同一对的 5 票方向完全一致，偏置被 5 票一致放大而不是抵消；而偏置是每个 pair 自己的系统偏移，Elo 又在整张 win matrix 上全局拟合，误差只会传播、不会对消。

现改为确定性双顺序：

- `_get_pair_sqls_to_eval`：每对产出 `(a, b, False)` 与 `(a, b, True)` 两条，不再随机。
- `win_matrix` 票维由 `budget` 改为 `2 * budget`，并用 `np.nan` 初始化；`_update_win_matrix` 新增 `vote_offset`，正序写 `0..budget-1`、倒序写 `budget..2*budget-1`，两个方向互不覆盖。
- `_compute_robust_win_matrix` 改为忽略 NaN 槽位求均值（解析失败少票时不再被当成 0 分拉低结果）。
- 两种顺序的均值与论文的成对差等价：`Δ = 2·mean − 1 ∈ {+1, 0, −1}`。偏置只在一种顺序出现时，两序互相矛盾 → 0.5（即 TIE）；两种顺序都判同一方赢 → 1。
- 代价：pairwise 调用量翻倍（K=2 时 5 → 10 次/题），但该阶段本就是全流程调用量最小的一环。

聚合行为的单测：纯位置偏置 → 0.5、两序一致判 a 赢 → 1.0、第二序少 2 票 → 仍为 1.0（NaN 被忽略）、两序全 TIE → 0.5。

仍未做（同属 ⑥）：每类结果只取一个代表参与两两比较（Eq.27 的代表分数广播），当前仍是 top-k 全体两两。

### 2.8 其它改动

- `_get_top_k_sql_candidates` 返回值由 `List[...]` 改为 `(candidates, meta)`，meta 提供 `valid_count` / `unique_result_count` / `max_class_size` / `fallback_count`（门控所需）。
- `runner/benchmark_execution.py` 适配新的二元返回。
- `shortcut_consistency_gap_threshold` 标注为 deprecated（字段保留，避免既有 toml 校验失败）。

---

## 三、验证

用 bird-dev 3 题快照（36 / 195 / 896）+ stub 投票（不产生真实 LLM 调用）验证：

- 单元：并列且支持相同 → 保留默认；支持 2 > 1 → 覆盖；无并列 → 不触发。
- 题 36（K=2，consistency 0.5 / 0.4167）：门控不触发（merged gap 0.4167），pairwise 仍必跑，6 列版胜出，与改造前结果一致。
- 合成并列场景（K=3，全部 1/3，三来源）：`selection_gate.triggered=true`、`supports=[1,2,2]`、覆盖 idx0→idx1，trace 中已无 `gap_shortcut`。
- 题 195 / 896：`unique_result_count=1`，仍走 `single_candidate`，行为不变。

真实 3 题端到端（含 evaluation 追踪）此前已跑通，EX = 66.67%（2/3），trace 事件量 852+。

双标签判定的验证：

- 解析器单测 6 例：`(C,I)→A`、`(I,C)→B`、`(C,C)→TIE`、`(I,I)→TIE`、legacy `"A"→A`、`"maybe"→None`，结果全部符合预期（`None` 情形输出一条预期的 ERROR 日志）。
- 真实 LLM 端到端（题 36，`qwen3.6-27b`，n=5）：模型返回
  `<result><sql1_judge>Incorrect</sql1_judge><sql2_judge>Correct</sql2_judge></result>`，格式完全符合新模板；5 票全 B，`rm_scores=[0.3772, 0.6228]`、`merged=[0.8772, 1.0394]`、`ranking=[1,0]`，`selection_gate.triggered=false`（merged gap 0.1622、rm gap 0.2455），最终仍选中 6 列版，与改造前结论一致。

---

## 四、已知问题与未做项

**已记录、未修复**：schema linking 通配符 bug —— `_extract_tables_and_columns` 把 SQL 中的裸 `*`（乘号 / `COUNT(*)` / `SELECT *`）当作表名通配符，pattern 退化为 `.*` 并 `fullmatch` 命中整库所有表，导致 reversed linker 产出的 SQL 链接出整库、gold 抽取失真（`table_recall=1.0` 不可信）。

**L4 仍未对齐的 3 处**：

- ② 裁判模型：prompt 侧已按论文 E.4 对齐（独立双标签、无 Top-1 先验、TIE 合法，见 2.7）；但默认仍回退到同源生成 LLM（self-judge），且环境内只有一个可用模型端点；要真正独立需提供第二个模型端点（`judge_llm` 已支持 DeepSeek），要复现论文的微调 pairwise RM 则需 GPU 训练。
- ⑤ `S_maj`：整行结果哈希 vs cell-level triple 频率平均。
- ⑥ pairwise 规模：两种展示顺序已按 Eq.25 补齐（见 2.7）；仍缺「每类结果只取一个代表参与比较 + 代表分数广播给类成员」（Eq.27），当前是 top-k 全体两两。

生成阶段（L1）不做优化（无 GPU / 无训练条件）；SQL 修复（L3）已基本对齐，且约 1/3 的 SQL 必须经过修复才能执行，属于地基环节。
