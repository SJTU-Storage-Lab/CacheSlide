"""Small offline training/calibration CLI with no native engine dependency."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .inputs import read_cases


def parser():
    root = argparse.ArgumentParser(prog="cacheslide-offline")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("train", "calibrate"):
        sub = commands.add_parser(name)
        sub.add_argument("--model", required=True)
        sub.add_argument("--input", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--seed", type=int, default=9)
        if name == "train":
            sub.add_argument("--steps", type=int, default=20)
            sub.add_argument("--rank", type=int, default=8)
            sub.add_argument("--max-positions", type=int, default=256)
            sub.add_argument("--query-chunk-size", type=int, default=64)
            sub.add_argument("--lr", type=float, default=0.001)
        else:
            sub.add_argument("--adapter", required=True)
            sub.add_argument("--max-elements", type=int, default=4_194_304)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    import torch

    torch.manual_seed(args.seed)
    cases = read_cases(args.input, require_plan=args.command == "calibrate")
    if args.command == "train":
        from .training import train_adapter

        trained = train_adapter(
            args.model,
            [c.token_ids for c in cases],
            args.output,
            steps=args.steps,
            lr=args.lr,
            device=args.device,
            rank=args.rank,
            max_positions=args.max_positions,
            query_chunk_size=args.query_chunk_size,
        )
        result = {**asdict(trained), "output": str(trained.output)}
    else:
        from .profiles import calibrate_profiles

        output = calibrate_profiles(
            args.model,
            args.adapter,
            [c.plan for c in cases],
            args.output,
            device=args.device,
            max_elements=args.max_elements,
        )
        result = {"output": str(output), "calibration_requests": len(cases)}
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
