"""SGLang commands; help and offline preparation never import native engines."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"train", "calibrate"}:
        from cacheslide_core.commands import main as offline

        return offline(args)
    if args and args[0] == "check-engine":
        parser = argparse.ArgumentParser(prog="cacheslide check-engine")
        parser.add_argument("--source-root")
        parsed = parser.parse_args(args[1:])
        from .compat import verify_compatibility

        print(json.dumps(verify_compatibility(parsed.source_root), indent=2))
        return 0
    if args and args[0] == "bench":
        args.pop(0)
    from .workflow import main as workflow

    return workflow(args)


if __name__ == "__main__":
    raise SystemExit(main())
