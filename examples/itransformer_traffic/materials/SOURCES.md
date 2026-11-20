# Traffic 示例的材料来源

| 本地文件 | 发布方与版本 | 用途 |
| --- | --- | --- |
| `traffic.csv` | 清华大学 THUML 的 [Time-Series-Library 数据集文件](https://huggingface.co/datasets/thuml/Time-Series-Library/blob/main/traffic/traffic.csv)。数据集标注 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。 | 按时间排列的 Traffic 数据。使用发布方提供的完整 CSV，保留原始字节。 |
| `itransformer_loader/data_provider/data_factory.py`、`data_loader.py`、`itransformer_loader/utils/timefeatures.py` | iTransformer 作者的[数据加载代码](https://github.com/thuml/iTransformer/tree/c2426e68ca13f74aaec08045c5c724d8ad328124/data_provider)，提交 `c2426e68ca13f74aaec08045c5c724d8ad328124`。版权及 MIT 许可随附于 `itransformer_loader/LICENSE`。 | 原样保留数据读取、滑窗和时间特征代码。入口为 `data_provider.data_factory.data_provider(args, flag)`。 |

Traffic CSV 发布大小为 136,478,119 字节，发布方记录的 SHA-256 为 `cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16`。可在 PowerShell 用 `Get-FileHash .\traffic.csv -Algorithm SHA256` 复核。

任务中的回看 96、标签段 48、预测 96 和 MSE 来自 [ReproduceBench 公开任务配置](https://github.com/AI9Stars/AutoReproduce/blob/e1a6c9220fadd9f6fae3b5e891eda7fe18989223/evaluation/exp_config.py#L36-L46)。本示例用 THUML 发布的数据和论文作者的加载器完成输入联调。与 [ReproduceBench 压缩包](https://huggingface.co/datasets/ai9stars/ReproduceBench)中数据及其单文件加载器是否逐字节一致，尚待核对。
