from axpy_ops import ops

from kernelkit import dim
from kernelkit.testing import op_tests

# the problems to test: two sizes x every dtype of axpy_problem's @dim
test_axpy = op_tests(ops.axpy, ops.axpy.problems(dim("n", (1000, 65536), id="n{}"), "dtype"))
