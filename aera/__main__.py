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
    else:
        print("usage: python -m aera {train|watch} [--help]")
        raise SystemExit(2)
    m(argv)


if __name__ == "__main__":
    main()
