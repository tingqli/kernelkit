"""``python -m kernelkit <cmd> <module>:<Tuning name> ...``: the tuner and table tools
(``kernelkit.tune``: problems, configs, tune, merge, show, verify, tune-all, compare) and
the A/B and correctness tools (``kernelkit.bench``: ab, check)."""

import sys

if sys.argv[1:2] in (["ab"], ["check"]):
    from .bench import main
else:
    from .tune import main

sys.exit(main())
