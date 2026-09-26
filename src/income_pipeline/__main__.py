"""Entry point for `python -m income_pipeline`."""

from __future__ import annotations

import sys

COMMANDS = ("generate", "train", "score")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("Income classification pipeline")
        print("  python -m income_pipeline generate --help")
        print("  python -m income_pipeline train --help")
        print("  python -m income_pipeline score --help")
        return
    if args[0] not in COMMANDS:
        print(f"unknown command: {args[0]}", file=sys.stderr)
        print("expected one of: generate, train, score", file=sys.stderr)
        raise SystemExit(2)
    if args[0] == "generate":
        from income_pipeline.generate import main as command
    elif args[0] == "train":
        from income_pipeline.train import main as command
    else:
        from income_pipeline.score import main as command
    command(args[1:])


if __name__ == "__main__":
    main()
