# AB-JChess

> A UCI Jieqi (揭棋) engine with an NNUE-style small network.

AB-JChess（引擎名 ABJchess，UCI 名称 `AB JChess 1 RB`）是揭棋（Jieqi，又称“暗棋”式中国象棋变体）的 UCI 引擎。
当前源码对应发布版 **AB-JChess-20261031**（Releases 中提供 Linux / Windows / macOS / Android 可执行文件与网络
`abjchess-20261030.nnue`）。

## 内容

| 目录 / 文件 | 说明 |
|---|---|
| `abjchess/` | 引擎源码（C++20）与多指令集 / 通用版构建脚本 |
| `abjchess-nnue-pytorch/` | 对应版本的网络训练器（PyTorch） |
| [编译指南.md](编译指南.md) | Linux、Windows、Android、macOS 编译与验证方法 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 第三方代码、版权与许可证声明 |

## 使用

AB-JChess 不含图形界面，请在支持 UCI 引擎的揭棋软件中加载。把网络文件 `abjchess-20261030.nnue`
放在可执行文件旁边（或当前工作目录），或用 UCI 选项指定：

```text
setoption name EvalFile value /path/to/abjchess-20261030.nnue
```

基本 UCI 命令：

```text
uci
isready
position fen <FEN>
go movetime 5000
```

FEN 采用 Pikafish 揭棋分支的格式（`X`/`x` 表示暗子）。验证：`bench` 应输出 `Nodes searched: 43125`。

> 发布包 AB-JChess-20261031 中的可执行文件由本源码编译，文件名为 `abjchess-*`，强制档位的环境变量为 `ABJ_TIER`。

## 编译

见 [编译指南.md](编译指南.md)。例如 Linux 通用版：

```sh
cd abjchess
JOBS=4 src/isa/build_universal.sh linux out-linux
```

## 训练

见 [abjchess-nnue-pytorch/README.md](abjchess-nnue-pytorch/README.md)。

## 致谢

- [Pikafish](https://github.com/official-pikafish/Pikafish) 与 [Stockfish](https://github.com/official-stockfish/Stockfish)：
  AB-JChess 早期版本基于 Pikafish，当前引擎仍含有派生自其的代码。
- [jqe](https://github.com/devinokeefe/jieqi)（Devin O'Keefe，MIT）：当前引擎以 jqe 为基础修改而来；训练器的网络结构为自研，设计上参考了 jqe。

## 许可证

AB-JChess 以 GNU General Public License v3.0 发布（见 `LICENSE`）。第三方声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
