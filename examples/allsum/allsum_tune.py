from allsum_ops import ops

from kernelkit.tune import Tuning

# how allsum is tuned: n is the bucket, tuned at three points (gather c4 cannot run 1001),
# on 2 and on 4 GPUs (world: a key of the table; each problem's ranks are started for it);
# every rank's x rotated through copies while timing
TUNING = (Tuning(ops.allsum, rotate=("x",))
          .set_bucket("n")
          .add_problem("allsum", n=(1001, 1 << 20, 1 << 24), world=[2, 4]))
