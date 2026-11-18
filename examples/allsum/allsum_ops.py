"""allsum: an op on several GPUs. Every rank holds its x; out = the sum of every rank's x,
on every rank (an all-reduce). The number of GPUs is the dim ``world`` of the problem (the
solvers are @multi_gpu("world")): `python -m kernelkit tune allsum_tune:TUNING allsum`
starts that many ranks itself."""

import os

import torch
import torch.distributed as dist

from kernelkit import Registry, knob, multi_gpu, no_capture

ops = Registry("demo-multi-gpu", table=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "allsum_tuned.csv"))
op_register = ops.register


# ---- the problem: every function runs on every rank ------------------------------------

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


@op_register
def allsum_check(out, expected, **inputs):
    """This rank's result alone (no collectives); a failure on any rank fails the config."""
    err = ((out.float() - expected).abs().max() / expected.abs().max()).item()
    assert out.dtype == inputs["x"].dtype and err < 1e-2, f"rel err {err:.1e}"
    return {"err": err}


# ---- the solvers: on the problem's world GPUs ------------------------------------------------

NO_GRAPH = "torch.distributed collectives: their CUDA-graph capture fails"


@op_register
@multi_gpu("world")
@no_capture(NO_GRAPH)
def allsum_rccl(x):
    out = x.clone()
    dist.all_reduce(out)
    return out


@op_register
@multi_gpu("world")
@knob("chunks", (1, 4), id="c{}")
@no_capture(NO_GRAPH)
def allsum_gather(x, *, chunks=1):
    """all_gather every rank's x, then add on each rank, in ``chunks`` pieces of equal
    size (a limit decided by the dims, so it raises on every rank)."""
    assert x.numel() % chunks == 0, f"n={x.numel()} % {chunks} chunks"
    out = torch.empty_like(x)
    for o, piece in zip(out.chunk(chunks), x.chunk(chunks)):
        parts = [torch.empty_like(piece) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, piece)
        torch.sum(torch.stack(parts), 0, out=o)
    return out
