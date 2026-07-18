# 项目全记录：《知道何时让位》—— 一份由浅入深的中文说明

> 论文标题（工作名）：**Knowing When to Yield: An Anatomy of Runtime Failure Alarms for Chunked Visuomotor Policies**
> （《知道何时让位：分块视觉运动策略运行时失败报警的解剖学研究》）
> 目标：AAAI-2027 ｜ 全部实验单卡 H200 完成 ｜ 代码+论文+数据全在 `/data2/zhu11/robotic/guard-testtime-recovery/`
>
> 本文档是**项目完整档案**，比 `overview.md` 更细，覆盖最终数据、每个实验、每次转向、以及"为什么这么做"。读者不需要机器人或机器学习背景也能读懂。

---

## 目录
1. 一句话与三句话
2. 背景知识（机器人怎么想、怎么动、怎么失败）
3. 领域现状与我们的切入点
4. 我们做的三件事（三个核心问题）
5. 方法：我们造了哪些"报警器"，怎么公平比
6. 五个最终发现（含真实数据表）
7. 研究历程：五次转向，跟着证据走
8. 为什么这是一篇"挑不出毛病"的论文
9. 我们踩过的坑（工程细节，供复现）
10. 文件导览与复现方式
11. 当前状态与后续

---

## 1. 一句话与三句话

**一句话**：机器人干活会突然搞砸，学界已经能装"失败报警器"了，但从没人系统研究过——报警之后到底能干什么；我们做了这个解剖，发现报警器**够用来"喊人接手"，但不够让机器人"自己修自己"**，并顺手挖出了现有报警器评测方法里的一个大坑和一条被忽视的规律。

**三句话**：
1. **评测有坑**：在"成功就提前结束"的任务上，一个只数"跑了多久"的傻瓜报警器判别力就能满分——现有累积式打分法虚高，我们给出干净的评测协议。
2. **信号有规律**：报警信号哪种最好，**取决于机器人的水平**——菜的机器人"走神走丢了"（分布外信号早早能抓），强的机器人"原地纠结"（只有一致性信号能抓，还只能事后抓）。这条规律解释了为什么各家论文结论互相打架。
3. **用途不对称**：报警能支撑"及时交接"（保留成功率 64%→72%），但不能支撑"重采样自救"（帮的和坑的一样多，统计上无差别）。

---

## 2. 背景知识

### 2.1 机器人怎么"想"和"动"
主流机器人操作 AI（**扩散策略 Diffusion Policy** 是代表）像一个每次算 16 步棋、只落 8 子就重算的棋手：看一眼画面 → 生成未来 16 步动作（一个"动作块"）→ 执行前 8 步 → 重新看、重新生成。它是**生成式**的：同一局面下心里有好几种合理做法（绕左推 / 绕右推），每次是随机抽一种，这叫**多模态**。

### 2.2 它怎么搞砸
- **误差滚雪球**：小偏差累积，把机器人带到没见过的状态，越偏越远。
- **模式跳变**：上一块走"绕左"，新一块抽中"绕右"，接缝处动作拧巴、左右摇摆而失败。这是生成式策略特有的病。

### 2.3 两个实验任务
- **Push-T**：推杆把 T 形积木推到指定位。官方模型成功率约 64%（约 1/3 失败，失败样本充足）。
- **robomimic Square**：机械臂抓方形螺母套柱子。失败结构完全不同（抓空、没对准），用来检验结论是否只在 Push-T 成立。

---

## 3. 领域现状与切入点

2024–2026 年，给策略装"运行时报警器"很热：
- **Sentinel / STAC**（CoRL 2024）：监控"这次想法 vs 上次想法"的分布差异，用成功轨迹的分位数定阈值；
- **FAIL-Detect**（RSS 2025）：把失败看成"走出舒适区"（分布外检测）；
- **FIPER / ActProbe / SAFE**：各种更强的学习型探针（但都需要训练/失败数据）。

**共同盲点**：这些工作都**报完警就停了**——没人在同一套受控协议下量化"报警到底支撑哪种下游决策"。这就是我们的切入点（这个措辞很关键：不是"没人研究过报警后怎么办"，而是"缺乏统一受控的下游量化"——因为 ARMADA、PATCH 等确实已在做交接/干预，我们如实承认并区分）。

---

## 4. 三个核心问题

- **Q1 报警器该监控"什么"？** 我们提出**边界一致性**信号：把"正在执行的计划后半段"和"新抽的想法们"比距离，取最近的那个——直觉是它锚定在真实执行的计划上。**它真更好吗？做了公平对决**。
- **Q2 报警器的"考试"出对题了吗？** 发现**时长泄漏**大坑，提出干净的**前缀协议**。
- **Q3 报警响了能干什么？** 两条路量化对比：**让位**（交给人/备用）vs **自救**（重采样恢复）。

---

## 5. 方法：报警器全家福 + 公平比法

**信号全家福**（都在同一批数据、同一套校准下比）：
| 信号 | 通俗解释 | 阵营 |
|---|---|---|
| 边界一致性（我们的） | 当前计划离"最近新想法"多远 | 执行锚定 |
| STAC 能量/MMD/Chamfer | 上批想法 vs 这批想法的分布差异（忠实复现，含 CUSUM 聚合） | 想法自比 |
| 离散度 | 这批想法互相散不散 | 瞬时不确定 |
| 嵌入 OOD | 当前画面离"成功经历库"多远 | 分布外 |
| **时长 null** | 只数跑了多久 | 泄漏对照 |

**校准（保形/conformal）**：拿几十条**成功**轨迹，取信号最大值的 90% 分位当报警线，数学上保证正常轨迹误报率 ≤ 10%（此范式沿用 Sentinel/FAIL-Detect，我们不冒领）。

**防挑刺纪律**：开发集（官方 seed）用来设计信号；另外 2 个自训 seed + robomimic 是**冻结配方后只跑一次**的确认集；3 seed × 2 任务 × 每 seed 300 回合；全部带自助置信区间；恢复实验用配对 McNemar 检验。

---

## 6. 五个最终发现（真实数据）

> **终版修订（第二轮对抗审稿后，07-06 下午）**：一位审稿代理用计算证明了更深一层的泄漏——
> **max 聚合本身也吃时长信息**（保持回合长度、随机重排逐步分数的"置换 null"就能到 ~0.64，
> 因为失败回合更长、抽到高分的机会更多）。这让终稿口径再收紧一档：强制度的 0.68–0.69 里
> 只有 +0.00~+0.11 是真信号（随 seed 波动）；弱制度的 OOD 才是全研究唯一的大幅真信号
> （超额 +0.21~+0.26）。故事从"信号排名依赖制度"深化为——**"可监控性本身依赖失败类型：
> 强策略的犹豫型失败对整个免训练信号族基本不可监控"**。恢复负结果也改为严谨表述：
> "无可检测收益（CI 跨 −13~+17pp，MDE 7–14pp）"，并补上 Square 跨任务复现
> （+0.7pp/+1.3pp，p≈1）。让位的价值如实重框为"**更会挑时机的超时器**"（部分选择性来自
> 时长信息，但比纯超时早一倍交接）。以下各节数字为置换 null 修订前的口径，结论方向不变。
>
> **第三轮外部意见落实（07-06 晚）**：①参考文献用 arXiv API 全面核修（外部 AI 指出的 3 处
> 作者错误全部属实并修复）；②补 FIPER-style（RND+ACE）与 ActProbe-style 基线——前者最高
> 0.80 没超过免训练家族，后者的 oracle 上界在 Square 仅 0.27–0.45（弱制度信号在观测空间不在
> 动作空间）；③时间分层置换 null（Square 上高达 0.98）证明 max 聚合指标不可拯救，前缀协议是
> 唯一干净度量；④失败模式自动分类：强制度失败大头"近失型"不可监控、罕见"振荡型"一致性可抓
> （0.77–0.92）——可监控性取决于失败构成；⑤**500 配对回合功效运行：+0.2pp、帮 95/坑 94、
> p=1.0、MDE 5.4pp——n=150 时的 +6pp 确认为噪声，负结果达到最强形态**；⑥timeout 全前沿
> 如实展示（保留成功率完胜但 94–97% 才报）；⑦校准鲁棒性 n=10→97 全程保守稳定；⑧标题收窄为
> Diffusion Policies、摘要重写、图拆三张、制度声明限定为"同任务受控消融为主证据"。
>
> **第四/五轮意见落实（07-08 凌晨）**：①弱 Push-T 升至 **n=3**（25.3/25.3/29.0%），终版规律：
> STAC-MMD 是弱 Push-T 唯一稳健超 null 的信号（0.801±0.050，超额 +0.13~+0.24 三 seed 一致）；
> OOD 早期性限定为 Square 特有；种子间信号异质性如实报告并以失败构成解释；②引言明写
> "not a new-detector paper" + 决策时刻评估论点；③新增学习型基线表与决策效用表（后备成功率
> ≳0.45/0.25 时报警优于全自主）；④分类学降级为描述性证据（未人工核验）；⑤补 Rewind-IL（真实
> 存在：conformal 块间差异 + 状态回溯——其设计恰预设了我们的重采样负结果）、RoboMonkey、
> Code-as-Monitor；⑥**新实验：固定时长对照**——关闭成功终止后所有信号塌到 0.40–0.59，一致性
> 甚至低于 0.5（任务后行为污染）→ 置换 null、时间分层 null、固定时长三种修复全部失败，
> **前缀协议是唯一干净度量**（协议论证闭环）；⑦忠实 BID 正对照在跑（强/弱 checkpoint 设定）；
> ⑧附录：基线保真度差异 + 协议全表。诚实拒绝：强 Square/LIBERO/VLA/真机（定位 DP-scale 诊断研究）。
> **终局实验（07-08）**：⑨**忠实 BID 正对照**：即使按原文配方（同训练 run 的 100k checkpoint 作弱参考、
> backward+forward 双准则），逐步重规划仍崩塌——seed A 上 BID 7.3% vs 朴素闭环 5.3% vs 后退基线
> 52.3%——"密集重规划灾难性的制度里，候选选择救不了它"（明确声明不与 BID 原文场景矛盾）。
> 至此两份外部意见的全部可执行项落实完毕，论文为投稿终版（正文 7 页 + 引用 + 附录）。
> **附录构建完成（07-08 下午）**：`supplement.pdf`（8 页，A–J 十节）——路线图/协议形式化/
> 全参数设置/逐 seed 全表（含 TIDE-like 信号补充）/时长混淆对照全表/基线保真度审计/让位全扫描+
> 效用热力图（盈亏平衡线）/修复全条件+MDE/分类学（**作者目检 17/20 一致**，样例网格图 4 张入附录）/
> 复现包清单。主文 6 处加附录引用。**新增吸收态对照**（成功后机器人冻结）：信号仍 0.32–0.59——
> 任务后策略被查询在操作域之外，自一致性语义失效 → 从另一角度支持"前缀协议是唯一自然评估"。


### 发现一：强策略上，一致性信号"集体撞天花板"，K=2 就够
判别失败的 AUROC（0.5=瞎猜，1.0=完美），按 **seed 级"均值±标准差"**报告（审稿加固：不同 seed
的回合不独立，合并做 bootstrap 会虚假缩小置信区间）：

| 信号 | 强策略(3 seed, 52–64%) | 弱 Push-T(25%) | 弱 Square(2 seed, 16–25%) |
|---|---|---|---|
| 边界一致性 | 0.687±0.039 | 0.600 | 0.467±0.094 |
| STAC-MMD | **0.692±0.054** | **0.851** | 0.633±0.046 |
| 离散度 | 0.585±0.013 | 0.371（反转!） | 0.245±0.074（反转!） |
| 嵌入 OOD | 0.567±0.072 | 0.711 | **0.910±0.027** |
| 时长 null | 1.000 | 1.000 | 1.000 |

强策略上一致性全家打平（我们的 ≈ STAC，论文里明说"我们的探针没赢，贡献在审计本身"——这是
审稿意见要求的诚实表述），且样本数从 16 降到 **2 性能不掉**（0.672@K2 vs 0.668@K16）——监控几乎免费。

### 发现二：时长泄漏——现有累积式评测虚高（方法学贡献）
时长 null 报警器 AUROC = **1.0**（全部 6 组数据上都是）；CUSUM 累积分数对每个信号——连"反转"的离散度——都刷到 0.77–1.0，大半是"跑得久=失败"的泄漏，不是真本事。我们提出**前缀协议**（第 W 步只用前 W 个值预测还在跑的回合），干净无泄漏。

### 发现三：检测 ≠ 预测（限定在我们测试的信号族）
前缀协议下，强策略上所有信号塌到 **0.54–0.64**，报警首次触发中位在回合进行到 **~50%** 处。现有一致性/OOD 类报警是"进行中探测器"不是"预报器"。（学习型预测器 FIPER/ActProbe 在此族之外，可能更早——已明确限定范围。）

### 发现四：信号排名依赖策略水平（升为共同主线）
弱策略把排名**整个翻转**（3 个弱策略、跨 2 个任务复现）：嵌入 OOD 冲到 0.71–0.93，且在 Square 的两个 seed 上都是全研究**唯一真正早期可预测**的信号（无泄漏前缀-W8 = 0.809 / 0.913）；离散度反转到低于瞎猜（0.19–0.37）。
**规律：失败可预测性取决于失败类型**——菜策略是"无能型失败"（走进不像成功的状态，OOD 早早可见）；强策略是"犹豫型失败"（舒适区内模式摇摆，只有一致性信号能抓且只能进行中抓）。**这条规律解释了检测文献为什么各家结论打架**（每篇的策略制度不同）。

### 发现五：报警能"让位"，不能"naive 自救"（用途不对称，实用价值最强）
**让位有效且防"平凡收益"质疑**——我们做了"早报-报准前沿"分析（外部意见要求的严格版）：
- 超时规则事后近乎满分，但在回合 **94–97%** 处才报，只省 1–2 步无用执行——几乎没用；
- 前缀规则最早（11% 处）但不准；随机无效；
- **校准报警是唯一占据有用中段的策略**，且在两个留出 seed 上复现：α=0.3–0.4 保留成功率 72–78%(dev,从 64%)、68–74%(seedA,从 52%)、65–70%(seedB,从 58%)，在 42–56% 处交接、每次失败省 17–20 步。
- 弱策略上角色对调（如发现四所料）：OOD 弃置主导——25%→38–40%（弱 Push-T）、25%→46%（Square seedA，31% 覆盖率时达 68%）、16%→44–50%（Square seedB）；OOD 修复（嵌入库与分位数校准分家，消除样本内偏差）后实际假弃置率基本贴合 conformal 目标。

**naive 自救无效**：报警触发"重采样挑最一致模式继续"这类**训练无关**恢复，在强(64%)弱(25%)基座、各阈值下都是帮的≈坑的（配对 McNemar p>0.1）；密集重规划直接崩塌 65%→28–35%。**严格限定**：仅指"naive 训练无关重采样族"，学习型恢复/重试/引导（如 To Err is Robotic、DynaGuide）是互补方向——我们的负结果恰好为这些"有向恢复"方法的动机提供了受控实证。

---

## 7. 研究历程：五次转向，跟着证据走

1. **v1（GUARD 自救系统）**：想做"校准触发+恢复菜单" → 触发器有效但恢复一再打脸（帮≈坑）；
2. **v2（检测器+让位）**：改主打检测+选择性让位 → 深挖发现 Sentinel/STAC 已做校准检测，正面比信号优劣的路也被堵（全家打平）；
3. **v3（解剖研究）**：把两次打脸变成贡献——评测有坑→修好再看→检测≠预测→那报警有什么用→能让位不能自救；
4. **外部意见加固**：收窄所有过度声明（"naive 重采样"、"测试的信号族"、"DP 规模诊断研究"），让位补决策曲线防平凡收益，泄漏改中性措辞，引用逐条核实（发现 REACH 是幻觉引用，排除）；
5. **确认集 + crop 坑**：自训策略只有 25%，查出根因是 lerobot 默认关了随机裁剪增广——带正确配置重训得到 52%/58% 的强策略；而 25% 的弱策略**因祸得福**成了"制度依赖"这条共同主线的证据。

**核心教训**：负结果不是失败，是没人替你踩过的坑；把坑标出来就是贡献。

---

## 8. 为什么挑不出毛病
1. 每个论断都有对照（信号比有忠实 STAC；泄漏有 null 基线；恢复负结果有配对检验+多基座；让位有超时/随机/前缀对照）；
2. 尺子先验过（复现官方 64% 才开跑）；
3. 开发/确认分离，防"信号是挑出来的"；
4. 负结果如实报告，各发现互相咬合成自洽故事；
5. 全部声明严格限定范围（DP 规模、免失败数据、无重训恢复、成功集校准的 conformal 语义）；
6. 单卡可复现（环境锁定 + 一键脚本）。

---

## 9. 我们踩过的坑（供复现）
- **LeRobot 0.4.4**：扁平命名空间（无 `.common`）；`crop_shape` 默认 `None` 会让 Push-T DP 只到 25%，**必须传 `[84,84]`**；老 checkpoint 要先 `migrate_policy_normalization`。
- **pymunk**：必须 `<7`（6.11.1），否则 gym-pusht 崩。
- **robosuite/mujoco**：`robosuite 1.4.1` + **`mujoco==2.3.7`**（3.x 会 `mj_fullM` 报错）；`egl_probe` 要 `cmake<4` + `--no-build-isolation`。
- **数据集**：robomimic v2.1 格式要转 v3.0；HF 匿名限流用 `hf-mirror.com` 镜像+退避；robosuite 离屏渲染图像**上下翻转**（实证 corr 0.987 判定）。
- **确定性**：`cudnn.benchmark=True` 有跨运行随机性，配对比较放同一进程内即可。

---

## 10. 文件导览与复现
```
guard-testtime-recovery/
├── overview.md / docs/project_summary.md  ← 通俗故事（本文件）
├── DESIGN.md            ← 设计决策+每次转向的证据（6a–6f 记录了全部 pivot）
├── PROGRESS.md          ← 任务时间线
├── paper/aaai/main.tex  ← AAAI 官方模板论文（可编译，数字已填，含表1+图1四联）
├── paper/aaai/references.bib ← 全部引用经 workflow 核实（REACH 幻觉已剔除）
├── envs/harness.py, robomimic_harness.py  ← 两个 benchmark 的评测框架（同接口）
├── guard/collect.py     ← 富日志采集（原始样本块落盘→所有信号离线可算）
├── experiments/
│   ├── collect_logs.py       ← 采集入口（--benchmark pusht|square）
│   ├── final_report.py       ← 冻结配方多 seed/多任务分析（出表+图数据）
│   ├── signal_lab.py         ← 信号实验室（仅开发集！）
│   ├── yield_analysis.py     ← 让位决策曲线（超时/随机/前缀对照）
│   ├── make_final_figures.py ← 四联主图
│   ├── recovery_compare.py / main_compare.py  ← 恢复负结果实验
│   └── detector_analysis.py  ← 单 run 全指标
├── results/final/       ← report.json, fig_main.png, table_signals.tex
├── results/logs_*/      ← 各制度 300 集富日志
├── scripts/night_pipeline.sh ← 无人值守流水线
└── setup_env.sh + requirements.freeze.txt ← 一键复现环境
```
**复现主结果**：`bash setup_env.sh` → 采集各 checkpoint 日志 → `python experiments/final_report.py --runs ...` → `make_final_figures.py`。

---

## 11. 当前状态与后续（2026-07-06 上午）
- ✅ 三个强 Push-T seed（52/58/64%）+ 两个弱制度（25%）数据齐全，五大发现全部有确认集数据支撑
- ✅ AAAI 论文正文写完并编译通过（表1+四联图+全部引用核实），数字已填
- ✅ 外部审稿意见全部落实，claim 全线收窄
- 🔄 进行中：robomimic Square 恢复负结果补测（跨任务验证）、square 第二 seed 训练、三审稿人对抗评审
- ⏭ 收尾：按对抗评审意见改论文 → 补 square 恢复数字 → 终稿

## 12. 第 7 轮审稿（针对 supplement）全部落实（2026-07-08 下午）

外部 AI 审稿人审查了 supplement.pdf，指出的问题已逐一修复：

**硬伤修复**
- **表格截断**（最重要）：原 64 行的 per-seed 大表塞在单个 `table*` 浮动体里无法跨页，PDF 底部直接把 Square weak B 的后 5 行信号裁掉了。根因还叠加了 `\resizebox{\textwidth}` 会把窄表**放大**、行高随之变高。修复：拆成 3 张按制度分组的表（强 Push-T / 弱 Push-T / Square），去掉 resizebox。已逐行验证 8 个 run 的 TIDE-like 行全部出现在 PDF 文本层。
- **"all six runs" 与 8 个 run 不符**：根因是 `audit_extras.py` 的 RUNS 列表漏了弱 Push-T w2/w3。把两个 run **追加在列表末尾**（保持 torch RNG 流不变→旧 6 个 run 的数字逐字节复现，已 diff 验证），重跑后 yield 表 40 行（8 run × 5 α）、taxonomy 表覆盖全部 6 个 Push-T run、时间分层 null 补齐 w2/w3。正文两处 "six" 改 "eight"。
- **report.json 又被 absorbing 单跑覆盖**（记忆里预警过的坑）：用全 8 run 命令重新生成，关键数字与第 6 轮表格核对一致。
- **正文弱制度 yield 数字改为跨 seed 诚实范围**：原来只用 w1（"25%→38–40%"），w2 峰值其实只有 33%；改为 "25–29% → 33–40%（三个弱 seed 各自最优工作点，coverage 0.42–0.57）"。

**其余修复**
- 修复条件名人类可读化（如 "Push-T strong (dev), committed, p90 — pre-registered main"），caption 里给出 committed/consensus/p_x 的精确定义（与 guard/controller.py 实现核对过）。
- Supplement 参考文献补全：STAC/Sentinel、FIPER、ActProbe、BID、conformal（Vovk+Angelopoulos）、LeRobot、DP、robomimic 全部入 bib，独立可读。
- 复现性章节重写："The **submitted** code/data package contains..."（AAAI 只认投稿时材料）+ 两层 manifest 表（ZIP 内 <2MB 实体清单 vs 尺寸超限但单命令可再生的 41GB checkpoints/217MB 富日志）+ 完整环境（Ubuntu 24.04/Python 3.10.20/PyTorch 2.10.0+cu128/LeRobot 0.4.4/robosuite 1.4.1/mujoco 2.3.7/H200 NVL 141GB）。
- **`make_release.sh`**（仓库根目录）：一键打包 `paper/aaai/supplementary_material.zip`（1.2MB、56 文件），与 manifest 表一致，已构建。
- Fixed-horizon/absorbing 对照降格为 "dev-seed sanity check"（正文承认 post-success 查询本身就是 OOD），不再作为主证据；主证据链 = 置换 null 族（全 8 run）+ prefix 协议。
- Taxonomy 核验句改为 "up to six per class"（oscillating 只有 5、other 只有 3，全部展示）。
- Utility heatmap 放大为整页宽 figure*；taxonomy 示例图重排为每行 3 个 episode（轨迹+曲线上下叠放），一类一整页图，可读性大幅提升。
- Yield 表 caption 注明表内 EU 为 fallback cost c=0（图 1 才扫 cost）。
- 删除了残留的旧版 `supplementary.pdf`（防止投稿时拿错文件）。

**产物状态**：supplement.pdf 12 页（原 8 页）编译零错误、零未定义引用；main.pdf 8 页同步重编译。全部表格行经 pdftotext 验证无截断。

**追加扩充（同日晚，应用户要求）**：把三块现成数据做成正式表格补进 supplement（现 **14 页、13 张自动生成表**）：
- **prefix-W 全扫描**（§D，4 张表）：8 run × 5 信号 × W∈{4..12} 全点位 + 每个 W 的合格人群表（n/失败数）。结论：对 W 不敏感，强制度相干族全程 0.49–0.64，Square OOD 全程 0.79–0.92——正文选 W=8 不是调出来的。
- **K-ablation**（§D，1 张表）：8 run × K∈{2,4,8,16} 带 CI。与 K=16 差距最大 0.022（中位 0.01）——负结果不能归咎于采样预算不足，部署监视器 K=2 就够。
- **校准鲁棒性**（§G，1 张表）：校准集下采样到 n∈{10,20,40,全集} 的实际 false-deferral（α=0.3，50 次重抽）。所有 run 都 ≤ 名义 0.3（保序保证保守成立）；n=20 与全集差 ≤0.06。弱 run 校准成功集只有 12–23 集，caption 已注明 n≥20 时饱和。
- 写表前先对数据核过每个声明（初稿 "±0.01" 被 w2 的 0.015 打脸后改为精确值）。ZIP 已重新打包（含新 supplement.pdf）。

## 13. 第 8 轮审稿（M1–M7 major revision）：文字/分析已完成，两个新实验在跑（2026-07-12）

这轮审稿质量最高（懂生存分析和统计规范），M1–M7 全部接受，两条用诚实替代方案处理。

**已落实（文字+分析层）**
- **M1 范围收窄**：摘要减半并给出 monitorability 的正式定义（= AUROC 对置换 null 的超额）；每个 headline claim 首次出现即带范围限定（准静态操作、单一架构、强制度仅 Push-T）
- **M2 表述统一**：全文 "no effect detectable above a 5.4pp MDE"
- **M3 学术脉络**：duration leakage = 生存分析的 informative censoring（Kalbfleisch & Prentice）；prefix 协议 = **landmarking**（Anderson 1983; van Houwelingen 2007）；prefix AUROC ≈ time-dependent ROC（Heagerty & Zheng）；补引 Conformal Policy Learning（arXiv 2311.01457，已 API 核实）
- **M4 降级**：Q1 重写为 "锚定执行中计划有帮助吗？——没有"，boundary coherence 正式降为 baseline
- **M5 统计规范**：Setup 里预先声明 primary signal（bcoh distmin max-agg），其余标 exploratory；deferral 工作点全部补 bootstrap CI（新脚本 `deferral_ci.py`）；**关键验证：64→72 的提升用配对 bootstrap 差值检验，三个 seed 全部显著（dev +8.4pp [+3.2,+13.9]）**——审稿人的质疑经受住了检验；Fig 2a prefix bar 补 ±1 s.d. 误差棒
- **M6 干预透明化**：supplement §H 加入伪代码级流程（k-means 聚类、medoid、committed/consensus 精确定义）+ 把现有 2×3 扫描定位为 mode-selection/threshold ablation
- **M7 预注册**：删掉 "pre-registered"，改为可核查的时间线陈述（探索性 n=150 扫描在先 → confirmatory 条件在采集前固定）
- **杂项**：monitor 家族 in/out-of-scope 表（tab:scope）；oracle 行改名 + data-snooping 注；2026 文献标 concurrent；64/64.3/65.3 基线差异解释（同一 checkpoint 不同评测集）；survivor base rate（36→38%）讨论；p_fb 现实范围；**wall-clock benchmark：K=2 监视开销仅 +3.5%（954→987ms），K=16 为 3.1×**（`bench_overhead.py`）
- **页数纪律**：加了这么多内容后通过针对性压缩（fixed-horizon 段、Related Work 冗余、Discussion）把正文收回**恰好 7 页**（引用从第 8 页开始）

**在跑的两个实验（无人值守）**
1. **Can-ph 强策略**（直击 M1）：`scripts/can_pipeline.sh` 自动完成 数据集下载→v3.0 转换→DP 训练（200k 步）→300 集采集；完成后跑 final_report 并入分析。若 SR 达 60–80% 即为第二任务的强制度证据，可放宽 M1 范围限定
2. **repair n=2000 扩展**（锁死 M2）：seeds 500–1999 的 1500 个新配对集，冻结阈值 0.1178（复用 power500）；完成后与原 500 集合并 → MDE 降到 ~2.7pp

**诚实拒绝**：事后补 OSF 注册（学术不诚实）；双盲独立标注（无第二标注人——用户本人可随时补做 20 分钟得 Cohen's κ）

## 14. 两个新实验落地并写进论文（2026-07-13）

**Can-ph 强策略（M1 的实验腿）——结果比预期更有意思**
- SR = **80.3%**（300 集），货真价实的第二任务强制度
- **相干信号在 Can 上是真的**：bcoh max-agg 0.866 vs 置换 null 0.74 = **+0.12 真实超额**（更严的时间分层 null 下仍 +0.04）；energy +0.07；但 MMD/dispersion/OOD 都在 null 或以下
- **早期可预测性没变**：prefix-W8 仍在 0.52–0.62 的强制度带内
- 科学解读：**"强制度不可监测"是 Push-T 的发现而不是定律——决定可监测性的是失败构成而非策略质量**。这不是打脸而是把论文的 §5.3 论点变得更锋利：MMD 在弱 Push-T 赢但在 Can 低于 null；相干在 Can 赢但在强 Push-T 归于 null。审稿人 M1 的担忧完全被数据证实是对的
- 附带红利：Can 上的 deferral 是全部 run 里最漂亮的（α=0.3：retained 80.3%→92.8%，handoff 在 19% 处）

**repair n=2000（M2 锁死）**
- 扩展 1500 集（冻结阈值、seeds 500–1999）独立复现 null：+0.3pp（282/278）
- 合并 n=2000：**+0.2pp，377 helped / 372 hurt，CI [−2.4, +2.9]，MDE 2.7pp，p=0.88**

**论文集成**：9-run 分析链全部重跑（旧 8 组数字逐项未变）；Table 1 加 Can 列；§5.1 新段 + 小节改题 "Strong Push-T"；§5.3 论点强化；全文 MDE 5.4→2.7、n=500→2000；supplement §C 加 Can、§H 加扩展时间线与 pooled 行；范围声明更新为"强制度覆盖三个 Push-T seed 和一个 Can seed"。正文经压缩仍**恰好 7 页**；supplement 15 页；全表无截断验证；ZIP 重打包。

**教训**：给表格加列必须同步改 tabular 列规格（`lccc`→`lcccc`），否则 12 个 "Extra alignment tab" 错误。

## 15. 第 9 轮审稿（3 条概念/分析建议）全部落实——无需新数据（2026-07-14）

审稿意见在 `paper/aaai/Knowing When to Yield.md`，三条都很有见地，用已有 rich logs 全部实现。

**建议 1：区分"最终时长"与"已经过时间"** — 纯概念澄清。审稿人指出置换 null 混淆了两个 duration 量：最终 episode 长度（由结局决定、决策时不可得）vs 已过运行时间（合法在线信号）。已在 §5.2 明确：prefix AUROC 衡量的正是"在已知 episode 跑了这么久之上"alarm 的增量判别力；duration-aware 策略的在线效用是另一个问题（§5.4）。

**建议 2：评估同时用 elapsed time + alarm history 的实用检测器** — 实质新分析（新脚本 `time_aware_detector.py`）。三个训练自由的在线检测器：timeout（纯 elapsed）、alarm（纯 running-max）、combined（OR 组合）。在匹配 coverage、combined 保持 timeout retained ±3pp 的约束下比较 handoff。
- **核心结果**：combined **从不更晚**，且提前量**随 alarm 判别力增长**——强 Push-T 提前 0.03–0.20，Can 0.15–0.23，弱/Square 0.2–0.8
- **深层洞察**：timeout 的最大非退化 coverage ≈ 成功率（成功 episode 短、失败 episode 长），所以对弱策略（SR 25%）timeout 在高 coverage 几乎无用，OOD alarm 才是唯一有效的早期检测器
- 直接回答审稿人"alarm 是否改善强 time-aware 检测器"=是的，且随 regime 变化 → 强化 §5.3 论点
- 新 supplement §G 段落 + Table 12

**建议 3：更好隔离 regime-dependent monitorability 的因果** — 收窄措辞 + 用 within-run taxonomy 支撑。审稿人正确指出跨 regime 比较同时改变多个因素（去 crop 同时改成功率/表示/视觉偏移敏感度；Square 改任务+能力；Can 单 seed）。已把 §5.3 结论从"tracks failure type"收窄为"**associated with** observed failure composition in the tested settings"，明确列出所有混淆，并指出 **within-run taxonomy 是唯一固定其他因素的比较**（同一 run 内不同失败类型可监测性不同）作为因果的干净证据。摘要/findings 的"tracks...rather than policy quality"软化为"depends on failure regime, not competence alone"（Can vs 强 Push-T 直接支撑此点）。

**产物**：正文经大量压缩仍**恰好 7 页**（图宽降到 0.82/0.82/0.83 + 多处行文精简）；supplement 15 页；ZIP 重打包含 time_aware.json；全表无截断验证。三条建议全部诚实落实，没有一条需要新采集。

## 16. 第 10 轮审稿（7 条 major + writing 清单）核心全部落实（2026-07-17）

这轮审稿抓到了一个真正的内部矛盾，并促成了两个新实验和一个新分析框架。

**#1 EU 模型自相矛盾（最重要）**：审稿人用我们自己的公式算出 timeout 严格支配我们主推的 alarm（0.856 vs 0.687）——因为 EU 里没有时间项，而全文对 alarm 的辩护恰恰是"它触发得早"。修复：新脚本 `utility_time.py` 给 EU 加执行成本项 −c_t·E[执行 replans]/R_ref。**诚实的结果**：c_t=0 时 timeout 处处占优（承认审稿人完全正确）；单独 alarm 只在信号真实处（Square c_t≈0.11–0.24、Can 高 p_fb）反超；combined(OR) 规则按构造永不劣于 timeout，其增益恰好出现在 alarm 有真实信号的制度（弱制度 +0.03–0.15），在强 Push-T 退化为纯 timeout——**检测结论在效用语言下的重述**。主文 utility 表重写 + 新段落 + supplement 新表。

**#4 60k 欠训练实验**：SR 24.0%，完整复现弱制度签名（MMD +0.21 p=0.002、dispersion −0.17、OOD +0.14 p=0.002），失败构成与 crop-weak 几乎一致——弱制度模式源于"弱"本身而非增强特有敏感性。审稿人要的解耦拿到了。

**#6 OOD-gated Square repair**：controller 实现在线 OOD gate（与离线检测器同 split 同 embedding），结果依然 null（+0.0pp committed；+4.0pp consensus p=0.18）——"用对信号也救不了 naive repair"，负结果更硬。

**#5 组合重加权**：利用 AUROC=Σw_t·AUROC_t 精确分解做反事实混合交换——构成解释相干族差距的 109%、MMD 的 41%、OOD 的 19%（强制度 76% near-miss vs 退化制度 26%）。taxonomy 从"描述性"升级为"定量分解"。另加结构性盲点段（near-miss 分数在成功分布内 + success-quantile 阈值 → 召回差是近定义性的）。

**#3 事实修正**：删掉被自己表格打脸的 "OOD uniquely carries early signal"（MMD 在 Square B prefix=0.843），并正面呈现审稿人发现的**反转异常**（Square 上 prefix>max，如相干 B 0.65 vs 0.40——失败策略卡在错误 mode 里反而更自洽），顺带确立"信号方向 dev seed 预定、永不按 run 翻转"的部署规则。

**#2 统计推断**：置换 p 值（500 shuffles，add-one）进 supplement 全表；Table 1 加 excess 括号列；Setup 声明多重性策略（只认族内跨 seed 复现的超额：MMD 4/4 退化 run p≤0.01，OOD 2/2 Square p≤0.002）；prefix 全表加 bootstrap CI；MDE 精确二项 CI 验证（[−2.5,+3.0] 与正态近似一致）；3×扩容采集（强 seeds 各 600 集）在跑，完成后以"仅 supplement 鲁棒性注"形式给更紧的 excess CI（正文数字不动）。

**#7 + writing**：early time-series classification 引用（Xing 2012）；"prefix 协议下 0.5 重新成为正确基线"+"长度匹配不可行"说明；审计清单 box（协议可采纳化）；摘要瘦身到 3 个数字 0 破折号；retained success 定义；69–77% deferral 可行性注；"Six weak runs 列了 5 个"的陈年笔误修正；Römer 在 PDF 渲染正常（审稿人的抽取工具 artifact，不改）。

**页数之战**：本轮净增内容 ~1 页,通过 fig_extras 整图下沉 supplement（其全部内容已有三张 supp 表覆盖）、scope 表移 supplement、图宽 0.72、约 20 处行文精简、`\enlargethispage` 微调，正文收回**恰好 7 页**。主文 8 页 / supplement 18 页 16 表，零错误零未定义引用。
