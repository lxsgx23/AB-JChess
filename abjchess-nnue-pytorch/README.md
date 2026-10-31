# ABJchess 网络训练器

训练 ABJchess 使用的 HNN 小网络（文件格式 JQHNNUE1）的 PyTorch 训练器，是训练 `abjchess-20261030.nnue`
所用的训练器版本（移除了内部注释；另外 `--teacher` 现在可直接使用发布的小网，见下文，其余代码未改动）。

## 文件

| 文件 | 说明 |
|---|---|
| `train_hnn.py` | HNN 小网络训练主程序（king-bucket 棋子-位置特征 + 暗子池阈值特征，ABJ 风格网络头，导出 `.hnn`） |
| `train_snn.py` | 教师网络（ABJNNUE 大网）读取、特征与公用训练组件（`train_hnn.py` 引用） |
| `train.py` | 训练记录（112 字节 TrainRecord）读取、流式加载等公用函数（来自 jqe，MIT） |
| `snn_ref.py` / `hnn_ref.py` | 与引擎整数推理逐位一致的 numpy 参考实现 |
| `jqt1_ref.py` / `jqt1_torch.py` | 可选的威胁特征（`--att-width`，默认关闭；发布的小网未使用） |
| `sk_scheme.py` | 可选的 layer-stack 选择方案（`--stack-scheme sk1`） |

## 环境

Python 3.10+，带 CUDA 的 PyTorch，NumPy：

```sh
pip install -r requirements.txt
```

## 用法

1. **生成训练数据**：用引擎的 `selfplay` 命令自对弈，输出 112 字节/条的 TrainRecord 二进制文件，例如

   ```sh
   ./abjchess selfplay games 1000 nodes 5000 threads 8 out data.bin
   ```

   （其他参数：`depth`、`seed`、`random`、`hash`、`maxply`、`evalfile` 等，见 `abjchess/src/selfplay.cpp`。）
   另外准备一份不参与训练的数据作为验证集。

2. **训练**：

   ```sh
   python3 train_hnn.py data1.bin data2.bin ... --val-file val.bin --teacher abjchess-20261030.nnue --out runs/mynet
   ```

   - `--teacher`：教师网络，可直接使用发布包中的小网 `abjchess-20261030.nnue`（压缩格式会自动用 `../abjchess/tools/jqnz_tool.cpp`
     编译出的工具解压，需要 g++；也可用环境变量 `ABJ_JQNZ_TOOL` 指定已编译的 `jqnz_tool`，或直接给出解压后的 `.hnn`）。
     教师值用与引擎逐位一致的整数推理（`hnn_ref.py`，CPU）计算。也接受 ABJNNUE 格式的大网（`--init-ls` 只支持大网）。
   - 目标为 `wt*教师值 + ws*搜索分 + wr*对局结果`（在胜率空间，权重由 `--wt/--ws/--wr` 指定）；`--wt 0` 时可以不给 `--teacher`。
   - 没有 GPU 时加 `--device cpu`（仅适合小规模试验）。
   - 网络结构：`--width`（FT 宽度）、`--kb-scheme`、`--l1/--l2`、`--stack-scheme`、`--factor`。
   - 训练过程：`--epochs`、`--lr`、`--batch`、`--qat-from`、`--stream N`（数据大于内存时分块流式读取）等，完整列表见 `python3 train_hnn.py -h`。

3. **使用结果**：`runs/mynet/best.hnn`（或 `last.hnn`）可直接在引擎中 `setoption name EvalFile value .../best.hnn`，
   或用 `abjchess/tools/jqnz_tool z best.hnn mynet.nnue` 压缩。

## 许可证与致谢

训练器以 GNU GPL v3 发布（仓库根目录 `LICENSE`）。网络结构为自研，设计上参考了 jqe
（https://github.com/devinokeefe/jieqi ，Copyright (c) 2026 Devin O'Keefe，MIT）；`train.py` 为 jqe 原文件，
`train_hnn.py`、`train_snn.py` 沿用了其中部分数据读取代码，MIT 声明见 `LICENSE-jqe-MIT.txt`。感谢 Devin O'Keefe。
详见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
