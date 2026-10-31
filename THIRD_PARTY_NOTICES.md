# 第三方声明 / Third-party notices

本仓库整体以 **GNU General Public License v3.0**（见根目录 `LICENSE`）发布。引擎中含有 GPL 代码，
因此 ABJchess 引擎（`abjchess/`）作为整体按 GPLv3 发布；网络训练器（`abjchess-nnue-pytorch/`）同样按 GPLv3 发布。
下列第三方代码/作品的版权声明与许可证文本均予以保留。

This repository is distributed under the GNU GPL v3.0 (root `LICENSE`). The ABJchess engine contains GPL code
and is therefore distributed as a whole under GPLv3; the trainer is distributed under GPLv3 as well.
All third-party copyright notices and licence texts below are retained.

## 1. Stockfish / Pikafish / AB-JChess（GPLv3）

- 上游：AB-JChess（https://github.com/lxsgx23/AB-JChess）派生自 Pikafish（https://github.com/official-pikafish/Pikafish），
  Pikafish 派生自 Stockfish（https://github.com/official-stockfish/Stockfish）。
- 版权：`Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)`；Pikafish 作者名单见 `abjchess/AUTHORS`
  （其中包括 the Stockfish developers，https://github.com/official-stockfish/Stockfish/blob/master/AUTHORS）。
- 许可证：GNU GPL v3（`LICENSE`、`abjchess/Copying.txt`）。
- 在 ABJchess 引擎中的用途：评估后处理（`src/hnnue.cpp` 的 `runtime_value`：材料缩放、无吃子步数衰减、截断）、
  走子记号解析（`src/position.cpp` 的 `apply_token`，含暗子身份后缀）、UCI 标识与 AB JChess 网络头部结构
  （ABJ-style head：pairwise FT、16 个 layer stack、PSQT 桶）。这些文件保留 Stockfish 版权头。
  Pikafish 风格的 FEN 方言、128 位 bitboard 布局及 Stockfish 的 ARCH 名称（ISA 档位命名）、LEB128 网络容器格式亦沿用其约定。

## 2. jqe（MIT）

- 上游：jqe, an engine for Jieqi —— https://github.com/devinokeefe/jieqi
- 版权：`Copyright (c) 2026 Devin O'Keefe`
- 许可证：MIT，全文见 `abjchess/LICENSE-jqe-MIT.txt` 与 `abjchess-nnue-pytorch/LICENSE-jqe-MIT.txt`（亦附于本文末尾）。
- 引擎：ABJchess 的搜索框架（含暗子机会节点的 alpha-beta 搜索）、局面/走法生成、置换表、自对弈数据生成
  （`selfplay.cpp`，112 字节 TrainRecord）、UCI 循环等以 jqe 引擎（`engine/src/`）为基础修改而来。
- 训练器：网络结构为自研，设计上参考了 jqe；`train.py` 为 jqe 的 `engine/train/train.py` 原样文件（MIT），
  `train_hnn.py`、`train_snn.py` 调用并部分沿用了其中的数据读取/流式加载代码。
- 致谢：感谢 Devin O'Keefe 公开 jqe 引擎与训练代码。

## 3. 其他

- `src/jqmath.h`、`src/jqmath_tables.inc`：为保证各平台浮点结果逐位一致，`log`/`exp` 按 GNU C Library 2.41
  （`sysdeps/ieee754/dbl-64/e_log.c`、`e_exp.c` 的 FMA 版本）逐指令转写，并含其 `__log_data`/`__exp_data` 常数表。
  glibc 以 GNU LGPL v2.1 或更高版本发布（与 GPLv3 兼容）；版权归 Free Software Foundation, Inc. 及各贡献者。
- `src/jqmath.h` 中的软件 FMA 移植自 musl libc `src/math/fma.c`（Szabolcs Nagy），MIT 许可证，
  `Copyright © 2005-2020 Rich Felker, et al.`。
- 训练器依赖 PyTorch 与 NumPy（各自许可证，未随本仓库分发）。
- 本仓库的旧版训练器（基于 official-stockfish/nnue-pytorch，GPLv3）不是 `abjchess-20261030.nnue` 的训练器；
  若保留在仓库中，其 GPLv3 声明不变。

---

### jqe LICENSE (MIT)

```
MIT License

Copyright (c) 2026 Devin O'Keefe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
