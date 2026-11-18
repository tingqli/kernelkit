"""``python -m kernelkit <cmd> <module>:<Tuning name> ...``: the tuner and table tools
(``kernelkit.tune``: problems, configs, tune, merge, show, verify, tune-all, compare),
the A/B and correctness tools (``kernelkit.bench``: ab, check) and the table viewer
(``kernelkit.plot``: ``python -m kernelkit plot TABLE.csv``)."""

import sys

if sys.argv[1:2] in (["ab"], ["check"]):
    from .bench import main
elif sys.argv[1:2] == ["plot"]:
    from .plot import main
else:
    from .tune import main

sys.exit(main())
