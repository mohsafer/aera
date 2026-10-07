"""python -m aera train|watch ..."""
from __future__ import annotations

import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    argv = sys.argv[2:]
    if cmd == "train":
        from .scripts.train import main as m
    elif cmd == "watch":
        from .scripts.watch import main as m
    elif cmd == "sb3":
        from .scripts.sb3 import main as m
    elif cmd == "plot":
        from .scripts.plot import main as m
    elif cmd == "compare":
        from .scripts.compare import main as m
    else:
        print("usage: python -m aera {train|watch|sb3|plot|compare} [--help]")
        raise SystemExit(2)
    m(argv)


if __name__ == "__main__":
    main()
