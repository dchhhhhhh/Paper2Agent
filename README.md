# Paper2Agent 第一次开发验收

本轮只处理 iTransformer 在 Traffic 上的 96 步多变量预测。给定论文 PDF、实验说明、数据和加载器，用户要能用一条命令确认材料彼此匹配，并拿到一个真实训练批次。缺少材料时，程序返回具体字段，补齐后可以用同一张任务卡重试。

## 运行示例

在本 README 所在目录运行。示例已填写相对于任务卡的路径，从其他目录启动时也可传入任务卡的绝对路径。

```powershell
python -X utf8 check_task.py examples/itransformer_traffic/task.json
python -X utf8 -m unittest discover -s tests -v
```

本地验证使用 Python 3.12。新环境可先执行 `python -m pip install -r requirements.txt`。这一阶段使用 CPU，无需配置模型服务。若要保存结果，在第一条命令末尾加 `--report 本次检查.json` 并使用新的文件名。标准输出是 JSON，加载器的提示写入标准错误。

## 任务范围与示例输入

| 项目 | 约定 |
| --- | --- |
| 目标 | Traffic，多变量预测。观察过去 96 个小时，标签包含 48 小时的已知历史和 96 小时的未来值。后续正式评价指标为测试集 MSE。 |
| 输入 | 任务卡 `examples/itransformer_traffic/task.json`。其中 `paper_path`、`data_path`、`dataloader_path` 分别指向论文、CSV 和受信任的本地加载器源码入口。 |
| 本轮输出 | 一份输入问题清单，或一份训练批次记录，包括训练窗口数、张量形状、时间轴、数据有限性和历史重叠检查。 |
| 完成边界 | `train_batch_checked` 仅表示数据接口已接通。论文语义分析、模型生成、训练权重、测试 MSE 和方法忠实度属于后续运行。 |

任务设置取自 [ReproduceBench 官方实验说明](https://github.com/AI9Stars/AutoReproduce/blob/e1a6c9220fadd9f6fae3b5e891eda7fe18989223/evaluation/exp_config.py#L36-L46)。本轮使用论文作者公开的 Traffic 单文件与数据加载器。这条数据来源可独立核验；与 ReproduceBench 整包内的文件逐项比对仍待开展。

### 示例材料来源

| 本地文件 | 核验依据 |
| --- | --- |
| `materials/paper.pdf` | [论文 v4](https://arxiv.org/abs/2310.06625v4)，25 页。SHA-256 为 `83e889795af52c672090255493266e3ea93e7b2a32c528f58074f2f9f757ffc5`。 |
| `materials/traffic.csv` | [作者团队发布的 Traffic CSV](https://huggingface.co/datasets/thuml/Time-Series-Library/blob/main/traffic/traffic.csv)。完整文件的 SHA-256 为 `cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16`。 |
| `materials/itransformer_loader/` | 原仓库的 `data_factory.py`、`data_loader.py`、`utils/timefeatures.py` 和 MIT 许可证。未引入模型或训练答案。见 [作者仓库固定版本](https://github.com/thuml/iTransformer/tree/c2426e68ca13f74aaec08045c5c724d8ad328124)。 |

表中的材料路径均相对于 `examples/itransformer_traffic/`。可用 PowerShell 的 `Get-FileHash -Algorithm SHA256 文件名` 检查下载内容。

当前目录随附了完整 CSV。若通过 Git 将项目移交给另一位同学，CSV 因体积较大已列入忽略规则；可从[作者的单文件下载入口](https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv?download=true)恢复到任务卡中的路径。加载器版本与许可见 [材料说明](examples/itransformer_traffic/materials/SOURCES.md)。

## 模块怎样交接

| 模块 | 接收 | 返回 | 实现状态 |
| --- | --- | --- | --- |
| 任务输入检查 | JSON 任务卡及三个本地路径 | 校验后的任务对象，或按字段列出的缺失、不可读和设置问题 | `task_input.load_task` 已实现 |
| 数据加载和批次检查 | 任务对象；用作者接口 `data_provider(args, "train")` 取得数据集和加载器 | 一个批次的形状、历史对齐和有限性检查结果 | `train_batch.inspect_training_batch` 已实现 |
| 流程控制 | 输入检查结果、批次检查结果 | 含任务编号、阶段、状态、问题或批次摘要的记录 | `check_task.check` 已实现 |
| 论文分析与代码生成 | 论文、目标实验、已确认的设置 | 带原文出处和待确认项的实验规格，再按规格生成可运行工程 | 下一步拟接入 |
| 执行与实验验证 | 生成工程、数据、冻结的配置和预算 | 退出状态、日志、预测与真值、指标及产物路径，由控制程序决定推进或恢复 | 下一步拟接入 |

预检失败时不导入训练依赖。输入缺失或超出范围返回退出码 2，对应 `waiting_for_input` 或 `invalid_task`。加载器异常或窗口不匹配返回 1，对应 `batch_check_failed`。检查通过返回 0。模型调用数在本阶段固定为 0，入口本身没有连接模型服务。

## 两人怎样验收

1. **给定完整示例，运行检查命令。** 确认 `split=train`，输入的三个维度依次是批次、时间、变量。小批次大小 2、随机种子 0 都是本次联调设置。作者加载器按时间切分、只用训练段拟合标准化参数，再从训练窗口抽样。
2. **检查标签的时间含义。** `label_shape` 的时间长度应为 144。程序逐元素核对标签前 48 步与输入末 48 步，并核对它们的时间特征。只有后 96 步是预测目标，记录在 `forecast_target_shape` 中。
3. **给定缺文件的任务，运行同一入口。** 自动测试使用临时任务卡指向不存在的数据文件，检查程序在 `input_check` 阶段点名 `data_path`。测试还覆盖旧版任务卡的空路径，以及把预测长度改为 192 的情形。

本次实测使用完整 CSV。训练段共有 12,089 个可用窗口，本次读取其中一个批次。可直接查看[真实训练批次记录](evidence/first_train_batch.json)和[缺数据记录](evidence/missing_data.json)，也可在队友的环境重新运行上面的命令核对。

## 第二步　论文分析、生成和实际训练

从本目录执行下列命令。第一条通过已登录的 Codex CLI 发送任务规格与公开论文选页，生成实验规格，在独立复核通过后生成 PyTorch 模型。第二条在训练前检查论文的关键机制，随后进行训练。第三条从预测和按时间索引保存的真值独立重算 MSE。新实验使用新的运行目录，便于保留每次生成和训练记录。

```powershell
python run_experiment.py generate --card examples/itransformer_traffic/task.json --config examples/itransformer_traffic/cpu_first_run.json --run runs/my_traffic96
python run_experiment.py train --card examples/itransformer_traffic/task.json --run runs/my_traffic96
python verify_run.py --out runs/my_traffic96/fit --card examples/itransformer_traffic/task.json --report runs/my_traffic96/verification.json
```

准备生成环境需要已登录的 `codex` CLI 和 `pypdf`，训练与复核使用项目的 Python 依赖。两类依赖位于同一解释器时可用 `run_experiment.py all` 一次接通生成与运行。复用已生成模型时，可直接用 `experiment_runner.py --card ... --model ... --config ... --out 新目录` 从头训练。模型默认服务的账单金额取决于订阅或提供商，CLI 用量和实际可获得的金额分字段记录。

本次选择 CPU 首轮配置。模型宽度 32、两层两头、前馈宽度 64，使用 Adam 和均方误差训练 10 轮。每轮覆盖所有训练窗口，每个窗口随机抽取 172 个变量。每轮验证均用全部变量，按验证 MSE 选取检查点，冻结后在全部测试窗口及 862 个变量上计算标准化尺度的 MSE。固定种子、全部配置、生成源码、检查点和日志均保存在同一运行目录。

## iTransformer 方法核查与定向修复

新建运行默认使用 `--protocol reviewed`。独立复核者需要对照论文、任务卡、冻结配置，回答关于变量注意力、编码器和预测头、归一化、预测目标、本地实验设置的五个问题。程序核对引文是否出现在指定来源。需要改动时，保留初稿，只修订被指出的条目，再复核整份规格。依据不足、仍有冲突时，记录原因并停在规格阶段。

模型生成后，检查程序先读一个真实训练窗口，再用小张量检验变量之间的缩放注意力、两处残差的归一化顺序、传感器换序、样本独立性、传感器之间的作用和历史窗口归一化的还原。明确失败才会触发定向修复，最多两次，每次都重新跑全部检查；证据不足的候选留给人工复核。候选代码和对应结果在运行目录的 `quality/candidate_*/`，每次模型调用在 `calls/`。历史运行按原来的协议继续。

只检查生成物时，可以执行下面两条命令中的第一条。退出码 0 表示通过，2 表示未放行，系统或依赖出错会报告错误。进一步修复一个已诊断的失败时执行第二条，再重新检查。完整 `train` 或 `all` 会自动走完这一过程。生成环境和 PyTorch 环境分开时，将 `--python` 设为 PyTorch 解释器路径。

```powershell
python run_experiment.py check --card examples/itransformer_traffic/task.json --run runs/my_traffic96
python run_experiment.py repair --card examples/itransformer_traffic/task.json --run runs/my_traffic96
```

用 `--protocol baseline` 新建原流程对照。执行 `python compare_method_checks.py` 可重跑已知样例。独立生成对照、匿名评审规则及训练对象的选择见[实验方案](evidence/quality_study/PROTOCOL.md)。
