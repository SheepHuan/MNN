#!/usr/bin/env python3
"""GPU PMU sweep 调用平台无关 PMC Interpreter 的 CUDA 入口。"""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kernel_agent.pmc_interpreter.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
