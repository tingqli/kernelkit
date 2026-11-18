from axpy_ops import ops

from kernelkit.tune import Tuning

# how axpy is tuned: n is the bucket (matched inexactly by the dispatch), tuned at three
# points; dtype is not given: every value of axpy_problem's @dim (problems axpy-bf16 and
# axpy-fp32); x and y rotated through copies while timing
TUNING = (Tuning(ops.axpy, rotate=("x", "y"))
          .set_bucket("n")
          .add_problem("axpy", n=(1000, 4096, 1 << 20)))
