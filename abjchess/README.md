# ABJchess 引擎源码

ABJchess（UCI 名称 `AB JChess 1 RB`）是揭棋（Jieqi）UCI 引擎，对应发布版 **AB-JChess-20261031**
（默认网络 `abjchess-20261030.nnue`，bench 43125）。

- `src/`：引擎源码（C++20）。评估只使用小网络（HNN，FT 宽度 256）；网络文件不编进程序，随发布包单独提供。
- `src/isa/`：多指令集档位与“通用版”（运行时按 CPU 选择档位）构建脚本：
  `build_tier.sh`（单档位）、`build_universal.sh`（Linux / Windows / Android）、`build_macos.sh`（macOS 通用二进制）、
  `dispatch_*.cpp`（运行时选择）、`tiers.sh`（档位与编译选项表）、`macho_initsec.py`（macOS 构建辅助）。
- `tools/jqnz_tool.cpp`：网络文件压缩/解压工具。

编译方法见仓库根目录的 [编译指南.md](../编译指南.md)。

常用 UCI 选项：`EvalFile`（网络文件，默认 `abjchess-20261030.nnue`，先在程序所在目录、再在当前目录查找）、
`Hash`、`Threads`、`ChaseRule`（`none` / `default` / `skyrule`）。命令行参数会作为一条命令执行，例如 `abjchess bench`。

## 许可证与致谢

ABJchess 以 GNU GPL v3 发布（见 `Copying.txt` 与仓库根目录 `LICENSE`）。

- 含有派生自 AB-JChess / Pikafish / Stockfish 的代码（GPLv3，Copyright (C) 2004-2025 The Stockfish developers，作者名单见 `AUTHORS`）。
- 以 jqe（https://github.com/devinokeefe/jieqi ，Copyright (c) 2026 Devin O'Keefe，MIT 许可证，见 `LICENSE-jqe-MIT.txt`）为基础修改而来。感谢 Devin O'Keefe 公开 jqe。

详见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
