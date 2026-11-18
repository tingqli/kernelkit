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
