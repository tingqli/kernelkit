# kernelkit

把一个 GPU 算子（op）写成一组**同前缀的普通函数**，kernelkit 据此给出：

- **测试**：一行 `op_tests(...)` 生成 pytest 测试；
- **调优**：`python -m kernelkit tune ...` 在 GPU 上测每个配置，写出 CSV 调优表；
- **分派**：`ops.axpy(x, y)` 按调优表选最快的配置运行；
- **实验**：`python -m kernelkit ab / check ...` 按配置名做 A/B 计时和正确性检查；
- **多卡**：一次调用跨好几张卡的 solver 加 `@multi_gpu(8)` 或 `@multi_gpu("world")`（卡数是 problem 的 dim），以上命令自动起相应个数的进程运行（第 8 节）；
- **看表**：`python -m kernelkit plot <表>` 在浏览器里把调优表画成折线图，看出每个配置在哪个分桶区间最快（6.6 节）。

solver 用什么写都行（FlyDSL、Triton、HIP、torch……）。

[点击查看 tune csv 文件可视化页面](https://tingqli.github.io/kernelkit/kernelkit/plot.html)

目录：

1. [安装](#1-安装)
2. [上手：axpy](#2-上手axpy)
3. [写 op：函数约定](#3-写-op函数约定)
4. [调用：分派](#4-调用分派)
5. [测试：op_tests](#5-测试op_tests)
6. [调优：Tuning 与命令行](#6-调优tuning-与命令行)
7. [A/B 计时与正确性检查](#7-ab-计时与正确性检查)
8. [多卡 op：@multi_gpu](#8-多卡-opmulti_gpu)
9. [Python API 速查](#9-python-api-速查)
10. [新增一个 op 的清单](#10-新增一个-op-的清单)
11. [常见错误](#11-常见错误)

## 1. 安装

```bash
pip install "kernelkit @ git+https://github.com/tingqli/kernelkit.git@v0.2.0"
```

在其他项目的 `pyproject.toml` 里依赖它：

```toml
[project]
dependencies = ["kernelkit @ git+https://github.com/tingqli/kernelkit.git@v0.2.0"]
```

开发 kernelkit 本身：

```bash
git clone https://github.com/tingqli/kernelkit.git
pip install -e kernelkit
python -m pytest kernelkit/tests      # CPU 测试，不需要 GPU
```

| 模块 | 需要 |
|---|---|
| `kernelkit`（注册、分派、读表）、`kernelkit.table` | 只用标准库 |
| `kernelkit.testing` | pytest（默认的结果检查用 torch） |
| `kernelkit.timing`、`kernelkit.tune`、`kernelkit.bench`、`python -m kernelkit` | torch 和 GPU（ROCm 或 CUDA） |
| `kernelkit.spmd`（多卡 op 的进程和同步，第 8 节） | torch.distributed；没有 GPU 时用 gloo 在 CPU 上运行（测试用） |
| `kernelkit.plot`、`python -m kernelkit plot`（6.6 节） | 只用标准库；页面在浏览器里从 CDN 加载 Plotly |

kernelkit 不声明对 torch 的依赖，直接用环境里已经装好的 torch（不会替换掉 ROCm / CUDA 版的 torch）。

## 2. 上手：axpy

[examples/axpy](examples/axpy) 是一个可以直接运行的完整例子：op `axpy` 计算 `out = alpha * x + y`，有两个 solver，`torch`（一次 `torch.add`）和 `tiled`（每 `block` 个元素一次 add，最多 64 块）。

### 2.1 写 op：`axpy_ops.py`

```python
import os

import torch

from kernelkit import Registry, dim, knob, meta

ops = Registry("demo", table=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "axpy_tuned.csv"))
op_register = ops.register

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
MAX_TILES = 64


# ---- the problem ---------------------------------------------------------------------

@op_register
def axpy_ref(x, y, *, alpha=1.0):
    """The interface (x, y, alpha) and the expected answer (fp32)."""
    return alpha * x.float() + y.float()


@op_register
@dim("dtype", ("bf16", "fp32"), id="{}")
def axpy_problem(n, dtype="bf16", *, seed=0):
    """A problem (dims n, dtype) -> its inputs. seed: the inputs' random seed, not a dim."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, device="cuda", generator=g).to(DTYPES[dtype])
    y = torch.randn(n, device="cuda", generator=g).to(DTYPES[dtype])
    return dict(x=x, y=y, alpha=2.0)


@op_register
def axpy_dims(x, y, *, alpha=1.0):
    """Inputs -> the dims of a real call (and the place to check its arguments)."""
    assert x.shape == y.shape and x.dtype == y.dtype, (x.shape, x.dtype, y.shape, y.dtype)
    return dict(n=x.numel(), dtype={v: k for k, v in DTYPES.items()}[x.dtype])


@op_register
def axpy_check(out, expected, **inputs):
    """Raise when out is wrong; the returned metrics go into the table and the reports."""
    err = ((out.float() - expected).abs().max() / expected.abs().max()).item()
    assert out.dtype == inputs["x"].dtype and err < 1e-2, f"rel err {err:.1e}"
    return {"err": err}


# ---- the solvers -----------------------------------------------------------------------

@op_register
def axpy_torch(x, y, *, alpha=1.0):
    return torch.add(y, x, alpha=alpha)


@op_register
@knob("block", (1024, 16384), id="b{}")
@meta(tiles=lambda n, block: -(-n // block), cached_loads=False)
def axpy_tiled(x, y, *, alpha=1.0, block=1024):
    """One add per tile of ``block`` elements; at most MAX_TILES tiles (a limit: raise)."""
    assert -(-x.numel() // block) <= MAX_TILES, f"n={x.numel()}: > {MAX_TILES} tiles of {block}"
    out = torch.empty_like(x)
    for o, a, b in zip(out.split(block), x.split(block), y.split(block)):
        torch.add(b, a, alpha=alpha, out=o)
    return out
```

- `axpy_ref` 定义 op `axpy`：参数就是接口，返回值是参考答案。
- `axpy_problem` 的参数 `n`、`dtype` 是 **dim**（`seed` 除外），返回一个 problem 的输入。`@dim` 给 `dtype` 的取值命名。
- `axpy_dims` 从实际调用的输入读出 dim，分派用它查表。
- `axpy_check` 判断结果对不对，返回的误差写进表和报告。
- `axpy_torch`、`axpy_tiled` 是两个 **solver**；`block` 是 `tiled` 的 **knob**，取值由 `@knob` 给出。
- `tiled` 跑不了的问题（超过 64 块）直接 `assert`：调优时记为 n/a，不进表。

### 2.2 声明怎么调优：`axpy_tune.py`

```python
from axpy_ops import ops

from kernelkit.tune import Tuning

# how axpy is tuned: n is the bucket (matched inexactly by the dispatch), tuned at three
# points; dtype is not given: every value of axpy_problem's @dim (problems axpy-bf16 and
# axpy-fp32); x and y rotated through copies while timing
TUNING = (Tuning(ops.axpy, rotate=("x", "y"))
          .set_bucket("n")
          .add_problem("axpy", n=(1000, 4096, 1 << 20)))
```

### 2.3 写测试：`test_axpy.py`

```python
from axpy_ops import ops

from kernelkit import dim
from kernelkit.testing import op_tests

# the problems to test: two sizes x every dtype of axpy_problem's @dim
test_axpy = op_tests(ops.axpy, ops.axpy.problems(dim("n", (1000, 65536), id="n{}"), "dtype"))
```

### 2.4 运行

在 `examples/axpy` 目录里运行（`python -m` 会把当前目录加进 `sys.path`）。命令名之后是 `<模块>:<Tuning 变量名>`，再后面是 problem id 或组名。以下是 MI300X 上的输出，时间因机器而异。

```
$ cd examples/axpy
$ python -m kernelkit problems axpy_tune:TUNING                 # 有哪些 problem
axpy
  axpy-bf16 {'dtype': 'bf16'} n: 3 points 1000..1048576
  axpy-fp32 {'dtype': 'fp32'} n: 3 points 1000..1048576
$ python -m kernelkit configs axpy_tune:TUNING axpy-bf16        # 有哪些配置
torch
tiled b1024
tiled b16384
$ python -m kernelkit tune axpy_tune:TUNING axpy                # 调整组 axpy，写 axpy_tuned.csv
[07:34:50] axpy axpy-bf16 {'dtype': 'bf16'}: 3 configs, 8 copies of ['x', 'y']
[07:34:50] n   1000: tiled b1024 6.0 | tiled b16384 6.0 | torch 6.1
[07:34:50] n   4096: torch 5.9 | tiled b16384 5.9 | tiled b1024 14.2
[07:34:51] n 1048576: torch 6.4 | tiled b16384 127.8  (1 n/a: AssertionError)
[07:34:51] axpy axpy-fp32 {'dtype': 'fp32'}: 3 configs, 8 copies of ['x', 'y']
...
$ python -m kernelkit show axpy_tune:TUNING axpy-bf16           # 每个调优点最快的配置
| n | best | us | best torch | us | best tiled | us | capture pick |
|---:|---|---:|---|---:|---|---:|---|
| 1000 | tiled b1024 | 6.0 | torch | 6.1 | tiled b1024 | 6.0 |  |
| 4096 | torch | 5.9 | torch | 5.9 | tiled b16384 | 5.9 |  |
| 1048576 | torch | 6.4 | torch | 6.4 | tiled b16384 | 127.8 |  |
$ python -m kernelkit verify axpy_tune:TUNING axpy-bf16 -t 10 3000 2000000   # 经分派运行、检查、计时
[07:34:55] n     10 -> point   1000: tiled b1024                   5.9 us (table      6.0) | err 3.1e-03
[07:34:55] n   3000 -> point   4096: torch                         6.1 us (table      5.9) | err 3.4e-03
[07:34:55] n 2000000 -> point 1048576: torch                         7.4 us (table      6.4) | err 2.6e-03
$ python -m pytest -q test_axpy.py                               # 测试表里的配置
..............                                                           [100%]
14 passed in 1.56s
```

得到的调优表 `axpy_tuned.csv`（`dtype` 是 key，`~n` 是分桶列，`err` 来自 `axpy_check`）：

```
dtype,~n,config,us,err
bf16,1000,tiled b1024,6.0,2.2e-03
bf16,1000,tiled b16384,6.0,2.2e-03
bf16,1000,torch,6.1,2.2e-03
bf16,4096,torch,5.9,2.0e-03
...
bf16,1048576,torch,6.4,3.0e-03
bf16,1048576,tiled b16384,127.8,3.0e-03
fp32,1000,torch,5.9,0.0e+00
...
```

### 2.5 在代码里调用

```python
import torch
from axpy_ops import ops

x = torch.randn(3000, device="cuda", dtype=torch.bfloat16)
y = torch.randn_like(x)
out = ops.axpy(x, y, alpha=2.0)                          # 查表：n=3000 用调优点 4096 最快的配置
out = ops.axpy(x, y, alpha=2.0, config="tiled b1024")    # 指定配置（名字或 dict），不查表
ops.axpy.pick(dict(n=3000, dtype="bf16"))                # {'solver': 'torch'}：会运行哪个配置
```

## 3. 写 op：函数约定

### 3.1 术语

| 术语 | 含义 | 代码里 |
|---|---|---|
| **problem** | 一个具体问题 | 一组 dim 值（dict） |
| **dim** | problem 的一个维度（形状、dtype、路由方式……） | `<op>_problem` 的参数（`seed` 除外） |
| **inputs** | 一个 problem 的接口参数 | `<op>_problem` 的返回值、`<op>_ref` 的参数 |
| **solver** | 一种解法 | 函数 `<op>_<solver>`（`axpy_tiled` 是 solver `tiled`） |
| **knob** | solver 的旋钮 | solver 的仅限关键字参数，取值由 `@knob` 给出 |
| **config** | 一个 solver 加上每个 knob 的取值 | `{"solver": "tiled", "block": 1024}`，配置名 `"tiled b1024"` |
| **key** | 查表时精确匹配的 dim | 表头里以 dim 命名的列 |
| **bucket** | 查表时取 ≥ 调用值的第一个调优点的 dim（最多一个） | 表头里的 `~<dim>` 列 |

### 3.2 函数的角色

函数名决定角色。`<op>_ref` 定义 op，其余 `<op>_<后缀>` 归到已注册的、前缀最长的 op（`moe_dyn_ep3k` 在 `moe_dyn_ref` 注册后属于 `moe_dyn`）。

| 函数 | 必需 | 签名 | 作用 |
|---|---|---|---|
| `<op>_ref` | 是，最先注册 | 只有具名参数（不能有 `/`、`*args`、`**kwargs`，不能叫 `config` / `table`） | inputs → 参考答案；参数就是 op 的接口 |
| `<op>_problem` | 测试、调优需要；在 solver 之前注册 | 只有具名参数；`seed` 若有，写成 `*, seed=0` | dim 值 → inputs（dict） |
| `<op>_check` | 否 | `(out, expected, **inputs)` | 结果错就抛异常；可返回误差 dict（第一项是报告里的列）。默认 `torch.testing.assert_close` |
| `<op>_dims` | 分派需要 | 和 `<op>_ref` 相同的参数 | inputs → `{dim: 值}`；每次 `op(...)` 都调用它，也在这里检查参数 |
| `<op>_<solver>` | 至少一个 | 先是和 `<op>_ref` **完全相同**的参数，然后是带默认值的仅限关键字 knob，最后可选 `**over` | 计算结果 |

- 后缀 `ref`、`problem`、`check`、`dims` 是保留的；solver 名也不能以 `_problem` / `_check` / `_dims` 结尾。以 `_` 开头的函数不注册。
- solver 跑不了某个问题时自己抛异常（`assert` 写清原因即可）。不需要另外声明适用范围。
- 输入看不出来的 dim（例如 MoE 路由是均匀还是集中），由调用方作为关键字参数传入（见第 4 节的“提示”）。

### 3.3 标记

`@op_register` 写在**最外层**，标记写在它下面。

```python
@dim("act", ("silu", "gelu"), id="{}")                    # 给 dim 的取值命名（只用在 <op>_problem 上）
@dim("fp8", {"bf16": False, "fp8": True})                  # dict：key 就是 id
@dim("m, n", {"tp8": (4096, 256), "ep8": (4096, 2048)}, name="shape")   # 多个 dim 一起：dict + 轴名

@knob("bm", (32, 64), id="bm{}")                           # knob 的取值（只用在 solver 上）；多个 @knob 取笛卡尔积
@knob("split", lambda inter_dim: (1, 2) if inter_dim > 256 else (1,), id="ks{}")   # 取值可以是 dim 的函数

@meta(tile_m=lambda bm: bm, cached_loads=False)            # 给工具读的元数据：常量，或 dim / knob 的函数
@no_capture("passes a host counter")                       # 不能在 CUDA graph capture 中运行：capture 时分派跳过它
@multi_gpu(8, init=mori_init)                              # 一次调用跨 8 张卡，每张卡一个进程（第 8 节）
@multi_gpu("world")                                        # 卡数是 <op>_problem 的 dim world

space = stack(knob("cfg", ("a", "b"), id="{}"), meta(cached_loads=False))   # 几个标记合成一个，多个 solver 共用
```

- `id` 是只含一个 `{}` 的格式串（默认 `name + "{}"`），用在配置名和测试 id 里；取值打印后不能含空格或 `:`。
- `@meta(cached_loads=False)`：solver 不会通过 cache 读轮换的输入（第 7 节的副本数检查用它）。其他 `@meta` 项由使用方约定，用 `op.meta(cfg, name, problem)` 读。

### 3.4 注册

```python
ops = Registry("name", table=path_to_tuned_csv)   # 每个包一个，放在一个小模块里（例如 _ops.py）
op_register = ops.register                         # 注册函数的模块都 import 它
```

- 顺序：`<op>_ref` 最先；`<op>_problem` 在 solver 之前；嵌套的 op（`moe_dyn`）先注册 `moe_dyn_ref` 再注册它的 solver。
- 违反约定时注册抛 `TypeError`（第 11 节），op 保持不变。
- 读取：`ops.axpy` 或 `ops["axpy"]`；`Registry` 是 `{op 名: Op}` 的 Mapping。
- `table=None`：没有调优表，`op(...)` 必须传 `config=`，测试跑全部配置。

### 3.5 配置名

配置名是 solver 名加各 `@knob` 的 id，用空格连接：`"tiled b1024"`。带 `**over` 的 solver 可以在后面接覆盖项：`"ep3k m64:g2.pf=1,x=2"`。`op.name_of(cfg)` 打印，`op.parse(name)` 解析回 dict。

## 4. 调用：分派

```python
out = op(*inputs, **kw)                    # <op>_dims 读出 dim -> 查表 -> 运行最快的配置
out = op(*inputs, routing="hot")           # 关键字参数是 dim 名但不是接口参数：作为提示参与查表
out = op(*inputs, config="tiled b1024")    # 指定配置（名字或 dict），不查表（仍调用 <op>_dims 检查参数）
out = op(*inputs, table="other.csv")       # 用另一张表；table=None 表示不用表
```

查表规则：

- 表头里以 dim 命名的列是 **key**，精确匹配；`problem` 里为 None 的 key 按 `<op>_problem` 的默认值匹配。
- `~<dim>` 列是 **bucket**：取 ≥ 调用值的第一个调优点，比所有点都大时取最大的点。以调优点 `(1000, 4096, 1048576)` 为例：n ≤ 1000 用 1000 的行，1001–4096 用 4096，更大的都用 1048576。
- 同一个点上的行按 `us` 从快到慢，运行第一个；当前 stream 正在 CUDA graph capture 时跳过 `@no_capture` 的 solver。
- 只有**调优过的 key** 能分派；没调优的 key 抛 `ValueError`（先 `tune`，或传 `config=`）。
- 不属于这个 op 的 solver 的行会被跳过，几个 op 可以共用一张表。
- 表文件改了以后一秒内生效，不用重启进程。

`Op` 的方法：

| 方法 / 属性 | 说明 |
|---|---|
| `name`、`ref`、`problem`、`check`、`dims`、`solvers`、`dim_names`、`registry` | 名字、各角色的函数、`{solver 名: 函数}`、dim 名、所属注册表 |
| `problems(*axes)` | `[(id, problem)]`：各轴的笛卡尔积。轴是 `@dim` 的轴名（全部取值）、`(轴名, [id, ...])`（部分取值）或 `dim(...)`（临时的轴） |
| `configs(problem)` | 全部 solver 的全部配置 |
| `name_of(cfg)` / `parse(name)` | 配置 ↔ 配置名 |
| `run(cfg, *args, **kwargs)` | 用指定配置运行 |
| `meta(cfg, name, problem=None)` | 读 `@meta` 项 |
| `capturable(cfg)` | solver 没有 `@no_capture` |
| `pick(problem, table=, capturing=False)` | `op(...)` 会运行的配置；返回的 dict 是共享的，不要修改 |
| `tuned(problem, table=)` | 这个 problem 对应调优点上的全部配置，最快的在前 |
| `points(problem, table=)` | `[(分桶值, [(us, cfg), ...])]`：这个 key 的所有调优点 |

`problem` 参数是含全部 dim 名的 dict（未知的写 None），例如 `dict(n=3000, dtype="bf16")`。

```python
>>> op = ops.axpy
>>> op, op.dim_names
(Op('axpy', solvers=['torch', 'tiled']), ('n', 'dtype'))
>>> op.problems("dtype")
[('bf16', {'n': None, 'dtype': 'bf16'}), ('fp32', {'n': None, 'dtype': 'fp32'})]
>>> op.problems(("dtype", ["fp32"]), dim("n", (64, 128), id="n{}"))
[('fp32-n64', {'n': 64, 'dtype': 'fp32'}), ('fp32-n128', {'n': 128, 'dtype': 'fp32'})]
>>> [op.name_of(c) for c in op.configs(dict(n=4096, dtype="bf16"))]
['torch', 'tiled b1024', 'tiled b16384']
>>> op.parse("tiled b16384"), op.meta(op.parse("tiled b1024"), "tiles", dict(n=4096))
({'solver': 'tiled', 'block': 16384}, 4)
>>> [op.name_of(c) for c in op.tuned(dict(n=3000, dtype="bf16"))]
['torch', 'tiled b16384', 'tiled b1024']
>>> op(x, y, table=None)
ValueError: axpy: no tuned config for {'n': 3000, 'dtype': 'bf16'} (no table): tune it, or pass config=
```

## 5. 测试：op_tests

```python
from kernelkit import dim
from kernelkit.testing import op_tests

test_axpy = op_tests(ops.axpy, ops.axpy.problems(dim("n", (1000, 65536), id="n{}"), "dtype"))
```

`op_tests(op, problems, *, table=<注册表的表>)` 对每个 problem 生成：

- `<op>-<problem>-dims`：`<op>_dims(**inputs)` 和 problem 一致；
- `<op>-<problem>-<配置名>`：运行配置，用 `<op>_check` 和 `<op>_ref` 比较。有调优表时只测这个 problem 对应调优点上的配置（分派可能选到的）；表还不存在时只有 `-dims` 项；`table=None` 时测全部配置。

```
$ python -m pytest -q test_axpy.py --co
test_axpy.py::test_axpy[axpy-n1000-bf16-dims]
test_axpy.py::test_axpy[axpy-n1000-bf16-tiled b1024]
...
$ python -m pytest -q test_axpy.py -k "bf16 and tiled"
```

每个没有默认值的 dim 都要在 problem 里给值。多个 op 就写多个测试函数。跨卡的 op（第 8 节）生成的测试项都跳过（pytest 进程只是一个 rank），用 `python -m kernelkit check` 检查。

## 6. 调优：Tuning 与命令行

### 6.1 Tuning

```python
from kernelkit.tune import Tuning

TUNING = (Tuning(op, keys=None, rotate=(), table=<注册表的表>)
          .set_bucket(dim)                    # 先调用
          .add_problem(name, **dims)          # 一组 problem；可以调用多次
          .add_problem(...))
```

| 设置 | 含义 |
|---|---|
| `keys` | 表里存、分派时精确匹配的 dim。默认：除分桶维外的全部 dim。只能去掉确实不影响哪个配置最快的 dim |
| `rotate` | 计时时轮换副本的接口参数（通常是权重），让每次调用读不在 cache 里的数据 |
| `table` | `tune` / `merge` 写、`show` / `verify` 读的表。默认就是分派读的那张 |
| `set_bucket(dim)` | 分桶的 dim，写进表头为 `~<dim>` |
| `add_problem(name, **dims)` | 加一组 problem，`name` 是组名 |

`add_problem` 怎么展开：

- **给了的 dim**：list、tuple、range 是多个取值，其他是一个取值（取值本身是 tuple 时再包一层：`d=[(1, 2)]`）。
- **没给的 dim**：在 `<op>_problem` 的某个 `@dim` 上时取它的全部取值；否则取 `<op>_problem` 的默认值；都没有时报错。
- **分桶维**：给的取值就是这组的调优点；没给时用它的 `@dim` 取值。小值处要密，最大的点要覆盖实际会遇到的最大调用。
- 其余 dim 的取值做笛卡尔积。problem id = 组名，再用 `-` 接上每个有多个取值的 dim 的 id。只有一个 problem 的组，id 就是组名。

例：`add_problem("axpy", n=(1000, 4096, 1 << 20))` 没给 `dtype`，取 `@dim("dtype", ...)` 的全部取值，得到 `axpy-bf16`、`axpy-fp32`，各在 3 个点上调。

一个 shape 要长期调优和测试时，就用 `add_problem` 加进 `Tuning`；临时试一下用 `--set`（6.3）。

### 6.2 命令

```
python -m kernelkit <命令> <模块>:<Tuning 名> [problem 或组名 ...] [选项]
```

| 命令 | 用法 | 作用 |
|---|---|---|
| `problems` | `problems M:T` | 列出组、problem 和调优点 |
| `configs` | `configs M:T P [-t V]` | 列出全部配置 |
| `tune` | `tune M:T P [-t V ...] [--out T.csv]` | 在每个点上调优，合并进表（默认 Tuning 的表） |
| `merge` | `merge M:T A.csv B.csv ...` | 把若干张表合并进 Tuning 的表（或 `--table`） |
| `show` | `show M:T P [--table T]` | 每个调优点：最快的配置、每个 solver 最快的配置、capture 时的选择 |
| `verify` | `verify M:T P [-t V ...]` | 经 `op(...)` 分派运行（包括调优点之间的值），检查并计时，和表里的时间对比 |
| `tune-all` | `tune-all M:T [P ...] --gpus 0-7 --per-job 4 --out DIR` | 多卡调全部（或指定的）problem，合并成 `DIR/<表名>.new.csv` |
| `compare` | `compare OLD.csv NEW.csv` | 每个调优点新旧表的最快配置，以及沿用旧配置的代价 |
| `ab`、`check` | 见第 7 节 | |
| `plot` | `plot T.csv ... [--out F.html]`（或 `plot M:T`） | 在浏览器里看调优表（6.6） |

P 可以是 problem id 或组名（组里的每个 problem）。

| 选项 | 默认 | 含义 |
|---|---|---|
| `-t/--points V ...` | problem 的调优点（verify：1 3 20 100 300 1000 3000 10000） | 分桶值 |
| `--set NAME=VALUE ...` | | 覆盖 problem 的 dim 值（6.3） |
| `--table T` | Tuning 的表 | 读 / 写的表 |
| `--out` | `--table` | tune：输出表；tune-all：输出目录（默认 `/tmp/kit_tune`） |
| `--iters` | 20 | 每次计时的调用数 |
| `--rounds` | 3 | tune 的轮数：每轮新的输入，各配置交错计时，取中位数 |
| `--copies` | 8 | `rotate` 输入的副本数 |
| `--gpus` / `--per-job` | `0-7` / 4 | tune-all：用哪些卡、每个 problem 几张卡（跨卡的 op：`op.world(problem)`） |
| `--force` | | tune-all：跳过 GPU 空闲检查 |

几个常用组合：

```
$ python -m kernelkit tune axpy_tune:TUNING axpy-fp32 -t 4096 --out part.csv   # 只调一个点，写到另一个文件
[07:33:02] axpy axpy-fp32 {'dtype': 'fp32'}: 3 configs, 8 copies of ['x', 'y']
[07:33:02] n   4096: torch 6.0 | tiled b16384 6.0 | tiled b1024 12.5
$ python -m kernelkit merge axpy_tune:TUNING part.csv                          # 再合并进 Tuning 的表
merged 3 rows from 1 files into .../examples/axpy/axpy_tuned.csv
$ python -m kernelkit show axpy_tune:TUNING axpy                               # 组名：依次处理组里的每个 problem
```

tune 的日志每个点一行：最快的三个配置；`BAD`（`<op>_check` 失败）的原因；n/a（solver 抛异常）的个数和异常类型。BAD 和 n/a 的配置不进表。

### 6.3 `--set`：临时改 dim

`--set name=value` 覆盖 problem 的任意 dim（不能 set knob 或 `seed`）。分桶维总是取 `-t` 的值。

```
$ python -m kernelkit tune axpy_tune:TUNING axpy-bf16 --set dtype=fp32 -t 2048 --out set.csv
[07:33:09] axpy axpy-bf16 {'dtype': 'fp32'}: 3 configs, 8 copies of ['x', 'y']
[07:33:09] n   2048: torch 6.0 | tiled b16384 6.0 | tiled b1024 8.2
$ head -2 set.csv
dtype,~n,config,us,err
fp32,2048,torch,6.0,0.0e+00
$ python -m kernelkit tune axpy_tune:TUNING axpy-bf16 --set block=4
--set 'block=4': expected name=value, name in ['n', 'dtype']
```

- 只有 key 列进表：`--set` 一个不在 `keys` 里的 dim 只改变输入，不改变表里的 key。
- 用 `--set routing=hot` 调出来的行，只有 `op(..., routing="hot")` 这样带提示的调用才会选中（`<op>_dims` 读不出、调用时也没传的 dim 按 `<op>_problem` 的默认值匹配）。
- `configs`、`show`、`verify`、`tune-all`、`ab`、`check` 都接受 `--set`。

### 6.4 多卡：tune-all

```
$ python -m kernelkit tune-all axpy_tune:TUNING --gpus 0-1 --per-job 1 --out /tmp/kk/all
[07:33:57] axpy-bf16: GPUs [0], 3 points
[07:33:57] axpy-fp32: GPUs [1], 3 points
[07:34:03] all jobs done in 0.1 min
no BAD configs
merged 16 rows from 2 files into /tmp/kk/all/axpy_tuned.new.csv
next: python -m kernelkit compare .../axpy_tuned.csv /tmp/kk/all/axpy_tuned.new.csv
      cp /tmp/kk/all/axpy_tuned.new.csv .../axpy_tuned.csv
$ python -m kernelkit compare axpy_tuned.csv /tmp/kk/all/axpy_tuned.new.csv
dtype=bf16 ~n=1000  old tiled b1024    6.0  new tiled b1024    6.0  old-cfg-now    6.0  +0.0%
...
```

- 先检查 GPU 是否空闲（rocm-smi，GFX 使用率 > 10% 就退出；`--force` 跳过），再拿每张卡的 `/tmp/gpu-<i>.lock`（被别的任务占着就退出）。
- 每 `--per-job` 张卡一组，每组调一个 problem（组内第 g 张卡负责调优点的 `[g::per_job]`），空出来就接着做下一个。
- 跨卡的 op（第 8 节）：每个 problem 占 `op.world(problem)` 张空闲的卡，这些卡上的 rank 一起调它的全部调优点；卡不够时等前面的 problem 结束。
- 每个 problem 的输出和日志在 `DIR/<problem>_<g>.csv` / `.log`；最后合并成 `DIR/<表名>.new.csv`，确认后按提示 `cp` 覆盖旧表。

### 6.5 表文件

调优表是 CSV：key 列、分桶列 `~<dim>`、`config`，然后是 `us` 和 `<op>_check` 返回的指标。`config` 及之前的列是一行的身份：写入同身份的行会替换旧行。文件按身份排序，同一个点内最快的在前。可以手工编辑（删行、改时间），分派一秒内读到新内容。

### 6.6 看表：plot

`python -m kernelkit plot` 把调优表画成浏览器里的交互折线图（页面是 [kernelkit/plot.html](kernelkit/plot.html)，用 Plotly）：

- key 列（`~<dim>` 之前的列）取值相同的行是一**组**，每组一张图：横轴是分桶值，纵轴是 `us`，每个配置一条颜色不同的折线，图例给出配置和颜色的对应（一个配置在每组里颜色相同；同一 solver 的配置用同一种点形）。配置在某个点上没有行（n/a、BAD）时折线在那里断开。
- 图下的色带是每个分桶区间里**最快的配置**，也就是分派的选择：调用值落在 `(上一个调优点, 调优点]` 的用这个调优点上最快的配置，最后一个点也管比它大的值。下面的 Dispatch map 表按区间列出最快的配置、时间和它领先第二名多少（不到 1% 标黄：在噪声里）。
- **点图上的点**（或色带）弹出这一点的详情：时间、比最快慢多少、名次、`<op>_check` 的指标；这个点上所有配置的排名（点一行就换成那个配置）；这个配置在哪些区间最快、哪些区间离最快不到 2%、哪些点没有行；表里的整行（可复制）。
- 纵轴可以切换 `us` / `× fastest`（同一点上最快配置的倍数，最快的在 1 上）；x、y 各自 log / lin。
- 右边的图例：单击显示 / 隐藏，双击只看这一个（再双击恢复），悬停高亮，点 solver 名切换这个 solver 的全部配置；搜索框按名字过滤（空格分开的词都要出现）；All / Winners / Top 3 只看在某个点上最快 / 进前三的配置。隐藏了一些配置时，色带、`× fastest` 和名次只在显示的配置之间比较。
- 左边选组（每组下面的小色带是它的分派图）。↑/↓ 换组，←/→ 左右移动选中的点，Esc 关掉详情。视图（组、选中的点、坐标）写在地址栏里，刷新或复制链接后还是这个视图。

```
$ python -m kernelkit plot axpy_tuned.csv                 # 几张表：页面上方的标签切换；也可以写 <模块>:<Tuning 名>
kernelkit plot: http://127.0.0.1:8765/?csv=axpy_tuned.csv&live=1
  /axpy_tuned.csv <- .../examples/axpy/axpy_tuned.csv
  (each table is read again on every request; Ctrl-C to stop)
$ python -m kernelkit plot axpy_tuned.csv --out axpy.html # 一个自带表的 html 文件，可以直接发给别人
wrote axpy.html (1 table inside): open it in a browser
```

- 服务只用标准库，只绑定 127.0.0.1（`--host` 改），只提供页面和给出的表，每次请求都重新读表；页面每 2 秒重读一次（右上角的 Live），调优一边写表一边能看到新的点。端口默认 8765，占用时往后找。在远程机器上：在 VS Code 的 Ports 面板把这个端口转发到本地，再打开打印的链接。
- 不用服务也行：浏览器直接打开 `plot.html`，点 Open… 选本地 CSV（或拖进页面）；或者 `plot.html?csv=<url>`（`csv=` 可以写几个；对方服务器要允许跨域读取，例如 raw.githubusercontent.com；GitHub 的 blob 链接自动换成 raw 链接）；`?data=<CSV 文本>` 把表直接写在地址里。打开本地文件时 Copy link 把压缩后的表放进链接（`#data=z:...`）。
- 页面从 cdn.plot.ly 加载固定版本的 Plotly（带完整性校验），浏览器要能访问它；加载不了时只是不画图，组、图例和 Dispatch map 照常可用。

## 7. A/B 计时与正确性检查

```
python -m kernelkit ab    M:T P -t V [V ...] -v SPEC [SPEC ...] [选项]
python -m kernelkit check M:T P -t V [V ...] -v SPEC [SPEC ...] [选项]
```

problem、`--set`、输入都和调优相同（同一个 `Tuning`）。每个 `-t` 值作为分桶值；第一个变体是 base。

变体写法：

```
[label=]<配置名>[:k=v,...]        配置名，可加覆盖项（knob 或 **over 的项）
[label=]@table[:k=v,...]          每个值上调优表里最快的配置
[label=]py:<模块 | file.py>:<函数>  任意函数 func(**接口参数)，例如实验中的 kernel 副本
```

覆盖项的值：int、float、`true` / `false`、`none`；含 `/` 的转成 tuple（`a/b/c`）；其余是字符串。`label` 是表格里的名字，默认是整个写法。

```
$ python -m kernelkit ab axpy_tune:TUNING axpy-bf16 -t 4096 1000000 -v @table "tiled b16384" --rounds 3 --out exp
...
#### 1000000 n (seed 1000007)

| variant | config | us | delta % | rounds min-max | verdict | err ref | err base |
|---|---|---:|---:|---:|---|---:|---:|
| @table | `torch` | 6.1 | +0.0 | 6.1-6.1 | base | 2.9e-03 | 0 |
| @table#2 | `torch` | 6.1 | +0.0 | 6.1-6.2 | noise floor | 2.9e-03 | 0 |
| tiled b16384 | `tiled b16384` | 127.9 | +1984.7 | 127.7-129.0 | slower | 2.9e-03 | 0 |
| *noise band +-0.3%* |  |  |  |  |  |  |  |

#### summary: base in us, the others delta vs the base; ~ = within the noise band

| variant | config | 4096 | 1000000 |
|---|---|---:|---:|
| @table | `torch` | 5.9 us | 6.1 us |
| @table#2 | `torch` | +0.0%~ | +0.0%~ |
| tiled b16384 | `tiled b16384` | +0.0%~ | +1984.7% |

(written: exp/ab.md, exp/ab.jsonl)
```

- `ab`：每个变体先运行一次，和参考比较（失败记 BAD，抛异常记 FAIL，都会被丢弃并列在 dropped 里），再交错计时 `--rounds` 轮。base 再以 `<base>#2` 测一份，它和 base 的差就是噪声下限；只有超出噪声带才判 `faster` / `slower`，否则是 `~`。
- 副本数检查：`--copies` < 8 时，可能通过 cache 读轮换输入的变体（没有 `@meta(cached_loads=False)` 的 solver 和 `py:` 变体）会被拒绝计时。

```
$ python -m kernelkit check axpy_tune:TUNING axpy-fp32 -t 1000 1000000 -v @table "tiled b1024" "mine=py:mine.py:axpy" --graph --diff-tol 1e-6
| n | variant | config | err ref | err base | repeat | bitwise | graph | result |
|---:|---|---|---:|---:|---:|---|---|---|
| 1000 | @table | `torch` | 0 | 0 | 0 | - | 0 | OK |
| 1000 | tiled b1024 | `tiled b1024` | 0 | 0 | 0 | - | 0 | OK |
| 1000 | mine | `py:mine.py:axpy` | 0 | 0 | 0 | - | 0 | OK |
| 1000000 | @table | `torch` | 0 | 0 | 0 | - | 0 | OK |
| 1000000 | tiled b1024 | `tiled b1024` | - | - | - | - | - | FAIL AssertionError: n=1000000: > 64 tiles of 1024 |
| 1000000 | mine | `py:mine.py:axpy` | 0 | 0 | 0 | - | 0 | OK |

1 FAILED
```

- `check`：对参考（`err ref`）、对 base（`err base`）、重复调用和第一次的差（`repeat`）、`--bitwise` 时是否和 base 逐位相同、`--graph` 时 CUDA graph 回放和 eager 的差（`@no_capture` 记为 n/a）。`--diff-tol` 是 base / repeat / graph 允许的最大误差。有失败项时退出码为 1；`@table` 在没调优的值上记为 n/a，不算失败。

| 选项 | 默认 | 用于 |
|---|---|---|
| `-v/--variants` | `@table` | 变体 |
| `--seed` | 7 | 输入 seed = seed + 值（ab 第 s 个样本再加 1000 × s） |
| `--diff-tol` | | 对 base 的误差上限（超出：ab 标 DIFF，check 记失败） |
| `--bitwise` | | 也比较逐位相同 |
| `--copies` / `--rounds` / `--iters` / `--samples` | 8 / 5 / 16 / 1 | ab |
| `--no-dup` / `--no-ref` | | ab：不测 `<base>#2` / 不和参考比较 |
| `--repeats` / `--graph` | 5 / | check |
| `--out DIR` / `--name` | / 命令名 | 结果追加到 `DIR/<name>.md` 和 `DIR/<name>.jsonl`（第一行是运行信息：命令、git 版本、torch 版本、GPU、功耗上限） |

## 8. 多卡 op：@multi_gpu

有的 op 一次调用要用好几张卡，例如 EP 的 MoE 整层：dispatch、专家计算、combine 在各卡之间交换 token。这样的 op 每张卡一个进程（rank），每个 rank 用自己的输入调用它。在 solver 上加 `@multi_gpu(...)` 声明它跨卡运行，参数是卡数，或者给出卡数的 dim 名。整层的不同做法（一个 kernel 做完整层，或者通信库的 dispatch / combine 加单卡的专家 kernel）是同一个 op 的不同 solver，一起调优：

```python
@op_register
@dim("act", ("silu", "situv2"), id="{}")
def moe_ep_problem(tokens, experts_global=256, world=8, act="silu", *, seed=0):   # world 是 dim：表的 key
    experts = experts_global // world                                             # 每个 rank 的专家数由 dim 推出
    ...

@op_register
@multi_gpu("world", init=mori_init)  # 卡数 = problem 的 dim world；init：kernelkit 起的每个 rank 在第一次调用前执行一次
@knob("producers", (8, 16, 32), id="p{}")
def moe_ep_mega(x, W, topk_weights, topk_ids, *, comm, tokens, act="silu", situ=None, producers=16):
    ...  # 一个 persistent kernel 做完 dispatch、专家计算、combine

@op_register
@multi_gpu("world", init=mori_init)  # 一个 op 的 solver 写同样的 @multi_gpu
@knob("cfg", ("s32", "l128"), id="{}")
def moe_ep_mori(x, W, topk_weights, topk_ids, *, comm, tokens, act="silu", situ=None, cfg="l128"):
    ...  # MORI dispatch -> 专家 kernel -> MORI combine
```

- `@multi_gpu(8)`：卡数固定。`@multi_gpu("world")`：卡数是 `<op>_problem` 的 dim `world`，它和其他 dim 一样进表的 key，`--set world=4` 就在 4 张卡上运行。`op.world(problem)` 给出一个 problem 用几张卡（单卡 op 是 1），`op.multi_gpu` 表示 op 是否跨卡。
- 一个 op 的所有 solver 写同样的 `@multi_gpu`（不一致时注册报错）。
- 命令不变：`tune`、`verify`、`ab`、`check` 发现 op 跨卡，就自动按多卡方式运行。kernelkit 起 `op.world(problem)` 个进程运行同一条命令（一条命令里的 problem 卡数不同时，按卡数分成几个任务依次运行），rank r 用可见的第 r 张卡（用 `HIP_VISIBLE_DEVICES` 选卡）；每个 rank 先建 `torch.distributed` 进程组（主机之间 gloo，GPU 之间 NCCL / RCCL），再执行 solver 的 `init`。`tune-all` 给每个 problem 它需要的卡数（6.4）。`problems`、`configs`、`show`、`merge`、`compare` 不运行 op，和单卡一样。
- rank 0 的输出就是命令的输出；其他 rank 的输出写到临时目录里的 `rank<r>.log`，出错时打印出来。
- 运行时的分派不变：程序自己起好各个 rank（例如 sglang 的 8 个进程），每个 rank 调用 `ops.<op>(...)`，各自查表。
- `op_tests` 生成的测试项都跳过；用 `python -m kernelkit check` 检查。

### 8.1 函数约定

| 函数 | 每个 rank 上 | 约定 |
|---|---|---|
| `<op>_problem` | 用相同的 dim 和 seed 调用 | 返回本 rank 的输入（用 `torch.distributed.get_rank()` 区分）。卡数是 dim 时，它等于 `torch.distributed.get_world_size()` |
| `<op>_ref` | 都调用 | 返回本 rank 的参考答案；可以用集合通信（例如收集所有 rank 的 token） |
| `<op>_check` | 各自检查本 rank 的结果 | 不用集合通信；任一 rank 失败，这个配置在所有 rank 上都算 BAD；指标取各 rank 的最大值 |
| `<op>_dims` | 各自读 | 各 rank 必须相同，否则会选到不同的配置；卡数是 dim 时也要返回它（例如从进程组或 op 自己的对象读）。本 rank 的输入里看不出来的量（例如所有 rank 的 token 总数）由调用方作为提示传入；`verify` 检查各 rank 是否一致，分桶值不在 `<op>_dims` 里时也作为提示传入 |
| solver | 同时调用 | 跑不了时在所有 rank 上一起抛异常：只用各 rank 都知道的值（dim）判断，并且在第一个集合操作之前 |

### 8.2 各 rank 步调一致

所有 rank 必须以相同的顺序执行同样的集合操作。在结果可能因 rank 而异的地方，kernelkit 先汇总各 rank 的结果再统一决定：

- **配置的第一次调用**：所有 rank 都返回，才算能跑；所有 rank 都抛异常，记为 n/a；只有部分 rank 抛异常时，整个任务停止（退出码 3），因为其他 rank 可能在集合操作里一直等下去。前提是其他 rank 的这次调用能返回：GPU kernel 的 launch 和 NCCL 集合操作都立即返回；在主机上阻塞的集合操作（gloo，或 synchronize 之后再读结果）会一直等到进程组超时。
- **`<op>_check`**：汇总各 rank 的结果，任一失败就都失败。
- **计时**：每次调用前先过主机 barrier，再让 GPU 空转（主机在这期间 launch），再过设备 barrier（1 个元素的 all-reduce），所有 rank 同时开始；每次调用取最慢 rank 的时间。各 rank 得到相同的时间，排出相同的顺序。
- **写文件**：只有 rank 0 写表和报告。
- 其他地方的异常（只发生在某个 rank 上，例如重复调用时）：该 rank 退出，kernelkit 停掉其他 rank，打印出错 rank 的日志。

### 8.3 例子：allsum

[examples/allsum](examples/allsum)：每个 rank 有自己的 `x`，`out` = 所有 rank 的 `x` 之和（all-reduce）。卡数是 problem 的 dim `world`。两个 solver：`rccl`（一次 `all_reduce`）和 `gather`（`all_gather` 后在本卡相加，分 `chunks` 段，段数要整除 `n`）。

```python
@op_register
def allsum_ref(x):
    """The interface (this rank's x) and this rank's answer (fp32); collectives are fine."""
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x)
    return torch.stack(parts).float().sum(0)


@op_register
def allsum_problem(n, world=2, *, seed=0):
    """This rank's inputs (every rank gets the same dims and seed): n random bf16. world:
    the ranks, one per GPU."""
    g = torch.Generator(device="cuda").manual_seed(seed * world + dist.get_rank())
    return dict(x=torch.randn(n, device="cuda", generator=g).to(torch.bfloat16))


@op_register
def allsum_dims(x):
    """The same on every rank, so that every rank picks the same config."""
    return dict(n=x.numel(), world=dist.get_world_size())

# allsum_check: as for one GPU (no collectives)

@op_register
@multi_gpu("world")
@no_capture(NO_GRAPH)
def allsum_rccl(x):
    out = x.clone()
    dist.all_reduce(out)
    return out
```

`allsum_tune.py` 在 2 张卡和 4 张卡上各调一组：`add_problem("allsum", n=(1001, 1 << 20, 1 << 24), world=[2, 4])`。

```
$ cd examples/allsum
$ python -m kernelkit problems allsum_tune:TUNING
allsum
  allsum-world2 {'world': 2} n: 3 points 1001..16777216
  allsum-world4 {'world': 4} n: 3 points 1001..16777216
$ python -m kernelkit tune allsum_tune:TUNING allsum          # 命令和单卡一样；按卡数起两个任务
[kernelkit] allsum runs on 2 GPUs here (@multi_gpu): 2 ranks; rank 0 prints, the others write /tmp/kernelkit_ranks_68f74o3o/rank<r>.log
[11:05:06] allsum allsum-world2 {'world': 2}: 3 configs, 8 copies of ['x']
[11:05:07] n   1001: rccl 27.2 | gather c1 30.0  (1 n/a: AssertionError)
[11:05:07] n 1048576: rccl 76.1 | gather c1 84.3 | gather c4 191.8
[11:05:08] n 16777216: rccl 766.2 | gather c1 860.3 | gather c4 966.5
[kernelkit] allsum runs on 4 GPUs here (@multi_gpu): 4 ranks; rank 0 prints, the others write /tmp/kernelkit_ranks_hpr0wvvt/rank<r>.log
[11:05:24] allsum allsum-world4 {'world': 4}: 3 configs, 8 copies of ['x']
[11:05:25] n   1001: rccl 34.0 | gather c1 39.0  (1 n/a: AssertionError)
[11:05:25] n 1048576: rccl 63.9 | gather c1 100.5 | gather c4 232.6
[11:05:26] n 16777216: rccl 419.6 | gather c1 963.5 | gather c4 1099.6
$ head -3 allsum_tuned.csv                                      # world 是 key
world,~n,config,us,err
2,1001,rccl,27.2,3.6e-03
2,1001,gather c1,30.0,3.6e-03
$ python -m kernelkit check allsum_tune:TUNING allsum-world4 -t 1001 1048576 -v @table "gather c4"
...
| n | variant | config | err ref | err base | repeat | bitwise | graph | result |
|---:|---|---|---:|---:|---:|---|---|---|
| 1001 | @table | `rccl` | 4.4e-03 | 0 | 0 | - | - | OK |
| 1001 | gather c4 | `gather c4` | - | - | - | - | - | FAIL AssertionError: n=1001 % 4 chunks |
| 1048576 | @table | `rccl` | 4.6e-03 | 0 | 0 | - | - | OK |
| 1048576 | gather c4 | `gather c4` | 3.1e-03 | 6.1e-03 | 0 | - | - | OK |

1 FAILED
```

`gather c4` 在 n=1001 时所有 rank 都抛异常（段数不整除 n）：tune 记为 n/a，check 记为 FAIL，任务继续。时间是最慢 rank 的。

## 9. Python API 速查

```python
# 表文件（标准库）
from kernelkit import table
fields, rows = table.read(path)                 # 表头，行（{列: str}）
table.merge(path, rows, fields=None)            # 加入 / 替换行；文件不存在时创建
table.write(path, rows, fields)                 # 排序后原子替换
table.identity(fields); table.bucket(fields)    # 身份列；分桶 dim（"~n" -> "n"）
table.compare(old, new, label=None, order=None, out=print)   # 返回最快配置变了的点数

# 冷 cache 计时（torch）
from kernelkit.timing import copies, time_cold, Timer, interleaved, noise_band, verdict
import itertools
Ws = copies(W)                                  # 8 份副本（tensor，或 dict 里的 tensor），第一份是 W 本身
nxt = itertools.cycle(Ws).__next__
us = time_cold(lambda: fn(nxt()))               # GPU 时间中位数（us），不含 host 开销
res = interleaved({"a": fa, "a#2": fa, "b": fb}, Timer(Ws), rounds=5, iters=16)   # fn(W) 形式；ABBA 交错
band = noise_band(res["a"], res["a#2"])         # 噪声带（%）
verdict(delta_pct, band)                        # "~" / "faster" / "slower"

# 调优（torch）
from kernelkit import tune as kt
t = kt.resolve("axpy_tune:TUNING")
prob = t.problem("axpy-bf16", ["dtype=fp32"])   # {'n': None, 'dtype': 'fp32'}
kt.tune(t, prob, [2048], "out.csv")
kt.show(t, prob); kt.verify(t, prob, [10, 3000]); kt.merge(["a.csv", "b.csv"], t.table)

# A/B 与正确性（torch）：包装成自己包的工具时用
from kernelkit import bench as kb
kb.ab(t, prob, [4096], ["@table", "tiled b16384"], rounds=5, iters=16)
bad = kb.check(t, prob, [1000], ["@table", "tiled b1024"], graph=True)
rec = kb.Recorder(out_dir, "ab", kb.run_meta("ab", prog="python -m mypkg.bench", path=pkg_dir))

# 多卡 op（torch.distributed）
op.multi_gpu; op.world(problem)                 # 是否跨卡；一个 problem 用几张卡（单卡 op 是 1）
from kernelkit import spmd
spmd.rank(), spmd.world()                       # 本进程在 kernelkit 起的任务里的 rank、rank 数（任务外：0、1）
spmd.gather(obj)                                # 各 rank 的 obj，按 rank 排列（任务外：[obj]）

# 看表（标准库，6.6）
from kernelkit import plot
html = plot.embed([("t_tuned.csv", text)])      # 自带表的页面：一个 html 文件
server, url = plot.serve(["t_tuned.csv"])       # 页面和表的 HTTP 服务（127.0.0.1）；server.serve_forever()
```

`kb.ab` / `kb.check` 的钩子：`inputs(value, seed) -> 接口参数 dict`（替换 `<op>_problem`）、`annotate(inputs, resolved) -> (标题后缀, 记录字段)`（ab）、`alias(base) -> 配置名`（变体的简写）、`unit`（标题里分桶值的单位名）、`describe`（开头打印的说明）。

## 10. 新增一个 op 的清单

1. 建一个注册表模块：`ops = Registry("<包名>", table=<包目录>/configs/<op>_tuned.csv)`，`op_register = ops.register`。
2. 按顺序写函数，每个都加 `@op_register`（最外层）：
   - `<op>_ref(接口参数)`：参考实现；
   - `<op>_problem(dim..., *, seed=0)`：生成输入；要按名字引用的 dim 加 `@dim`；
   - `<op>_check(out, expected, **inputs)`：判对错，返回误差 dict；
   - `<op>_dims(接口参数)`：从输入读出 dim；
   - 每个 solver `<op>_<名字>(接口参数, *, knob=默认值, ...)`，每个 knob 一个 `@knob`；跑不了的情况 `assert`。
3. 写 `Tuning`：`set_bucket(<连续变化的 dim>)`，每个要支持的 shape 一个 `add_problem(...)`，权重放进 `rotate`。
4. 写测试：`test_<op> = op_tests(ops.<op>, ops.<op>.problems(...))`。
5. 运行：`python -m kernelkit configs ...` 看配置 → `tune`（或多卡 `tune-all` + `compare` + `cp`）→ `show` / `verify` → `pytest`。
6. 调用方直接 `ops.<op>(...)`；没调优过的 key 会报错，补 `add_problem` 后重新调优。
7. 一次调用跨好几张卡的 op：每个 solver 加同样的 `@multi_gpu(卡数或 dim 名)`（rank 需要额外初始化时带 `init=`），problem / ref / check / dims 按 8.1 写。

## 11. 常见错误

| 报错 | 处理 |
|---|---|
| `@op_register must be the outermost decorator` | `@op_register` 写在所有标记的上面 |
| `no registered op is a prefix of the name; register its <op>_ref first` | 先注册 `<op>_ref` |
| `<op>_problem: register it before the solvers` | `<op>_problem` 写在 solver 前面 |
| `the parameters must start with exactly those of <op>_ref` | solver 的前几个参数和 ref 完全相同（名字、种类、默认值） |
| `each knob needs exactly one @knob` | 每个 knob 正好一个 `@knob` |
| `@dim describes the problem; it goes on <op>_problem` / `only @dim marks` | `@dim` 只用在 `<op>_problem` 上，`@knob` 只用在 solver 上 |
| `seed ... must be keyword-only with a default` | 写成 `def <op>_problem(..., *, seed=0)` |
| `dim(...) needs name=` | 多个 dim 一起的 `@dim` 要给 `name=` |
| `knob(s) [...] clash with the dims` | knob 和 dim 不能同名 |
| `... prints as '...', which reads back as ...` / `cannot parse` | knob 取值打印后含空格或 `:`，或 id 区分不了不同取值 |
| `no @dim 'x' on <op>_problem` / `has no value [...]` | `op.problems` 引用了不存在的轴名或取值 id |
| `dims [...] unset` | 传给 `op_tests` 的 problem 缺少没有默认值的 dim |
| `no tuned config for {...} in <表>: tune it, or pass config=` | 这个 key 没调优过：`add_problem` 后 `tune`，或传 `config=` |
| `no tuned config for {...} can run under CUDA-graph capture` | 这个点上的配置都是 `@no_capture`：给能 capture 的 solver 调优 |
| `picking a config needs <op>_dims; pass config=` | 写 `<op>_dims`，或调用时传 `config=` |
| `unknown argument(s) [...]` | `op(...)` 的关键字参数既不是接口参数也不是 dim |
| `a tuned table has at most one bucket column` | 表头里只能有一个 `~` 列 |
| `not a kernelkit.tune.Tuning` | 命令名之后要写 `模块:Tuning 变量名` |
| `no tune problem or group 'x'` | 用 `problems` 命令查看可用的 problem id 和组名 |
| `call set_bucket first` / `call it before add_problem` | 先 `set_bucket`，再 `add_problem` |
| `give the points of the bucket` | 分桶维没有 `@dim`：在 `add_problem` 里给出调优点 |
| `give [...] (no ..._problem default, no @dim)` | 在 `add_problem` 里给出这些 dim |
| `[...] already name a group or a problem` | 组名和 problem id 重名：换一个组名 |
| `variants [...] may read the rotated inputs with cached loads` | 用 `--copies` ≥ 8；solver 确实不走 cache 读时加 `@meta(cached_loads=False)` |
| `GPUs busy (GFX use %)` / `GPU g is locked by another job` | 卡被占用：等一下或换 `--gpus`（`--force` 只跳过空闲检查，不跳过锁） |
| 两个调优点之间的调用失败 | 分桶维上的下限型限制（例如要求 `n >= block`）在点之间可能不满足：在 solver 内处理，或把这个 dim 作为 key |
| `f_k runs on 1 GPU(s), the solvers [...] of f on 8 GPU(s): every solver of an op runs on the same GPUs` | 给 op 的每个 solver 加同样的 `@multi_gpu` |
| `@multi_gpu('world'): not a dim of f_problem` | 卡数用 dim 名时，`<op>_problem` 要有这个参数，并且先注册 |
| `f runs on 8 GPUs here (@multi_gpu); 2 visible` | 让至少这么多张卡可见（`HIP_VISIBLE_DEVICES`） |
| `problems {...} run on more GPUs than --gpus` | tune-all 的 `--gpus` 给够卡 |
| `... raised on rank(s) [3] only ...; stopping the job`（退出码 3） | solver 只用各 rank 都有的值（dim）判断能不能跑，并在第一个集合操作之前抛异常 |
| `... (outside an agreed step: ...); stopping the job` | 只在某个 rank 上出现的异常（例如重复调用、CUDA graph capture 时）：看打印的 traceback 和该 rank 的日志 |
| `BAD <op>_dims differ between the ranks` | `<op>_dims` 只返回各 rank 相同的值，其余由调用方作为提示传入 |
