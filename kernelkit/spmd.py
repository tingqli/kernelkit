"""Multi-GPU jobs: an op whose solvers are ``@multi_gpu`` runs on ``world`` processes, one
per GPU (a rank each), every rank calling it with its own inputs. ``world`` is the
``@multi_gpu`` number, or the problem's value of the dim it names (``op.world(problem)``).

``python -m kernelkit tune / verify / ab / check`` on such an op start the ranks
themselves (``launch``; one job per number of GPUs among the command's problems):
``world`` processes of the same command, rank r on GPU r of the visible ones. Rank 0
prints as the command would; the other ranks write to ``<tmp>/rank<r>.log``, printed when
they fail. ``tune-all`` gives each problem ``world`` GPUs. A rank first joins the job
(``init``: ``torch.distributed`` over gloo for the hosts and NCCL / RCCL for the GPUs, then
the solvers' ``init`` functions), then runs the command in lockstep with the others:

* ``<op>_problem`` and ``<op>_ref`` run on every rank with the same dims and seed: the
  problem returns this rank's inputs (``torch.distributed.get_rank()``), the ref this
  rank's answer (it may use collectives, e.g. to gather every rank's tokens);
* the first call of a config is agreed (``first_call``): it returned on every rank, or it
  raised on every rank (the config cannot run the problem). Raising on some ranks only
  stops the job, as the other ranks may wait for them in collective work for ever: a
  solver decides whether it can run from values every rank has (the dims), before its
  first collective step. (The job stops once the other ranks' calls return, as GPU
  launches and NCCL collectives do at once; a call blocked on the host in a collective,
  e.g. over gloo or after a synchronize, waits until the process group's timeout.)
* ``<op>_check`` runs on each rank alone (no collectives); a failure on any rank fails
  the config on every rank, and the metrics are the worst rank's (``check``);
* a timed call starts on every rank together (host barrier, GPU spin while the hosts
  enqueue, device barrier) and takes the slowest rank's time (``kernelkit.timing``), so
  every rank ranks the configs alike;
* only rank 0 writes the table and the reports.

Any other exception in a rank stops the job: the rank exits, ``launch`` stops the others
(after ``GRACE`` seconds) and reports. Without GPUs (the tests) the ranks run on the CPU
over gloo.
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback

__all__ = ["ENV", "ABORT", "GRACE", "rank", "world", "child", "launch", "init", "finish",
           "gather", "first_call", "check", "slowest", "barrier", "align", "abort",
           "expect_agreed"]

ENV = "KERNELKIT_RANK"  # "<rank>/<world>" in the processes of a job
ABORT = 3  # exit code of a rank that stopped the job
GRACE = 10.0  # s the other ranks have to exit after one failed, before they are stopped
_TAIL = 40  # lines of a failed rank's log printed

_rank, _world, _job = 0, 1, False
_one = []  # the device barrier's tensor


def rank():
    """This process's rank in a job (0 outside one)."""
    return _rank


def world():
    """The number of ranks of this process's job (1 outside one)."""
    return _world


def child():
    """(rank, world) when this process was started as a rank (``ENV``), else None."""
    v = os.environ.get(ENV)
    if not v:
        return None
    r, _, w = v.partition("/")
    return int(r), int(w)


# ---- the launcher ----------------------------------------------------------------------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stop(procs):
    for p in procs:
        p.terminate()
    end = time.monotonic() + 5
    for p in procs:
        try:
            p.wait(max(0.0, end - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def _tail(path, n=_TAIL):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError as e:
        return f"({e})"


def launch(argv, op, world, *, log=print):
    """Run ``python -m kernelkit <argv>`` as the ``world`` ranks of a job of ``op`` (see the
    module docstring) -> its exit code: the ranks' when they agree, else the code of the
    rank that failed first."""
    import torch

    n = world
    gpus = torch.cuda.device_count()
    if gpus and gpus < n:
        vis = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
        raise SystemExit(f"{op.name} runs on {n} GPUs here (@multi_gpu); {gpus} visible "
                         f"(HIP_VISIBLE_DEVICES={vis})")
    logs = tempfile.mkdtemp(prefix="kernelkit_ranks_")
    log(f"[kernelkit] {op.name} runs on {n} GPUs here (@multi_gpu): {n} ranks; rank 0 "
        f"prints, the others write {logs}/rank<r>.log")
    sys.stdout.flush()
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()))
    procs, files = [], []
    try:
        for r in range(n):
            out = None
            if r:
                out = open(os.path.join(logs, f"rank{r}.log"), "w")
                files.append(out)
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "kernelkit", *argv], env=dict(env, **{ENV: f"{r}/{n}"}),
                stdout=out, stderr=subprocess.STDOUT if out else None))
        codes, first, since, stopped = [None] * n, None, None, []
        while None in codes:
            for r, p in enumerate(procs):
                if codes[r] is None and p.poll() is not None:
                    codes[r] = p.returncode
                    if codes[r] and first is None:  # an aborting rank: the others abort too
                        first, since = r, time.monotonic() - (GRACE - 2 if codes[r] == ABORT
                                                              else 0)
            if first is not None and None in codes and time.monotonic() - since > GRACE:
                stopped = [r for r, c in enumerate(codes) if c is None]
                _stop([procs[r] for r in stopped])
                for r in stopped:
                    codes[r] = procs[r].returncode
            time.sleep(0.1)
    except BaseException:  # KeyboardInterrupt, ...: no rank outlives the launcher
        _stop([p for p in procs if p.poll() is None])
        raise
    finally:
        for f in files:
            f.close()
    if first is None:
        shutil.rmtree(logs, ignore_errors=True)
        return 0
    if stopped:
        log(f"[kernelkit] rank {first} exited with {codes[first]}: stopped rank(s) {stopped}")
    if len(set(codes)) > 1:  # the ranks disagree: show the failed ones
        for r, c in enumerate(codes):
            if c and r and r not in stopped:
                path = os.path.join(logs, f"rank{r}.log")
                log(f"[kernelkit] rank {r} exited with {c}; the end of {path}:\n{_tail(path)}")
    return codes[0] if len(set(codes)) == 1 else codes[first]


# ---- in a rank ---------------------------------------------------------------------------

def init(op, world):
    """Join the job of ``world`` ranks as the rank given by ``ENV``: the process group (gloo
    + NCCL; gloo alone without GPUs), then the ``init`` functions of ``op``'s solvers."""
    global _rank, _world, _job
    import torch
    import torch.distributed as dist

    r, n = child()
    if n != world:
        raise SystemExit(f"{ENV}={os.environ[ENV]}, but {op.name} runs on {world} GPUs here")
    if torch.cuda.is_available():
        torch.cuda.set_device(r)
        dist.init_process_group("cpu:gloo,cuda:nccl", rank=r, world_size=n,
                                device_id=torch.device("cuda", r))
    else:
        dist.init_process_group("gloo", rank=r, world_size=n)
    _rank, _world, _job = r, n, True
    for f in op._inits():
        f()


def finish():
    """Leave the job at the end of a command (nothing outside a job)."""
    global _rank, _world, _job
    if not _job:
        return
    import torch
    import torch.distributed as dist

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    barrier()
    dist.destroy_process_group()
    _rank, _world, _job = 0, 1, False


def gather(obj):
    """Every rank's ``obj`` (picklable), in rank order (``[obj]`` outside a job)."""
    if _world == 1:
        return [obj]
    import torch.distributed as dist

    out = [None] * _world
    dist.all_gather_object(out, obj)
    return out


def first_call(fn, what):
    """``fn()``, the first call of a config, agreed between the ranks: its result when it
    returned on every rank; this rank's exception when it raised on every rank; when it
    raised on some ranks only, the job stops (see the module docstring)."""
    if _world == 1:
        return fn()
    try:
        res, err = fn(), None
    except Exception as ex:  # noqa: BLE001 - agreed below
        res, err = None, ex
    got = gather(None if err is None else f"{type(err).__name__}: {err}")
    failed = [r for r, e in enumerate(got) if e is not None]
    if not failed:
        return res
    if len(failed) < _world:
        abort(f"{what} raised on rank(s) {failed} only ({got[failed[0]][:300]}); a "
              "multi-GPU solver raises on every rank or on none (it decides from values "
              "every rank has, e.g. the dims, before its first collective step): the other "
              "ranks may wait for it in collective work")
    err.kit_agreed = True
    raise err


def check(fn):
    """``fn()``, a check of this rank's result (no collectives), agreed between the ranks:
    its metrics dict, each metric the largest (worst) of the ranks'. A failure on any rank
    raises on every rank: AssertionError (RuntimeError when some rank raised another
    exception), naming the rank."""
    if _world == 1:
        return fn()
    try:
        res, err = fn() or {}, None
    except Exception as ex:  # noqa: BLE001 - agreed below
        plain = isinstance(ex, AssertionError)
        res, err = None, (plain, str(ex) if plain else f"{type(ex).__name__}: {ex}")
    got = gather((res, err))
    fails = [(r, e) for r, (_, e) in enumerate(got) if e is not None]
    if fails:
        r, (plain, msg) = next(((r, e) for r, e in fails if not e[0]), fails[0])
        ex = (AssertionError if plain else RuntimeError)(f"rank {r}: {msg}")
        ex.kit_agreed = True
        raise ex
    merged = {}
    for m, _ in got:
        for k, v in m.items():
            merged[k] = max(merged[k], v) if k in merged else v
    return merged


def slowest(times):
    """The slowest rank's time of each timed call (``times``: this rank's, the same count
    on every rank)."""
    if _world == 1:
        return list(times)
    return [max(c) for c in zip(*gather(list(times)))]


def barrier():
    """Host barrier: every rank's host is here (gloo)."""
    import torch
    import torch.distributed as dist

    dist.all_reduce(torch.zeros(1))


def align():
    """Device barrier on the current stream (a 1-element NCCL all-reduce): what is
    enqueued after it starts on every GPU together."""
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        return
    if not _one:
        _one.append(torch.zeros(1, device="cuda"))
    dist.all_reduce(_one[0], op=dist.ReduceOp.MAX)


def abort(msg):
    """Stop the job: print ``msg``, exit this rank with ``ABORT`` (``launch`` stops the
    others)."""
    print(f"[kernelkit rank {_rank}] {msg}; stopping the job", file=sys.stderr, flush=True)
    sys.stdout.flush()
    os._exit(ABORT)


def expect_agreed(ex):
    """In a job, stop it unless ``ex`` was raised on every rank at once (by ``first_call``
    or ``check``): any other exception leaves the ranks at different steps."""
    if _world > 1 and not getattr(ex, "kit_agreed", False):
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        abort(f"{type(ex).__name__}: {ex} (outside an agreed step: the other ranks may wait "
              "for this one)")
