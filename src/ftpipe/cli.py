"""ftpipe command-line interface.

    python -m ftpipe.cli <command> [options]

Commands: prepare | train | sweep | evaluate | judge | forgetting | serve
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftpipe", description="QLoRA function-calling pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="build the dataset from the HuggingFace Hub")
    p.add_argument("--config", default="configs/data.yaml")

    p = sub.add_parser("train", help="fine-tune with the given config")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--run-name", default=None)

    p = sub.add_parser("sweep", help="run the hyperparameter sweep")
    p.add_argument("--config", default="configs/train.yaml")

    p = sub.add_parser("evaluate", help="evaluate a model on the benchmarks")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--tag", default="base", choices=["base", "finetuned"])

    p = sub.add_parser("judge", help="LLM-as-judge: base vs fine-tuned")
    p.add_argument("--config", default="configs/eval.yaml")

    p = sub.add_parser("forgetting", help="catastrophic-forgetting check")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", default=None)

    p = sub.add_parser("serve", help="start the FastAPI server")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "prepare":
        from .config import DataConfig, load_config
        from .data import prepare

        prepare(load_config(args.config, DataConfig))

    elif args.command == "train":
        from .train import main as train_main

        train_main(args.config, args.run_name)

    elif args.command == "sweep":
        from .sweep import main as sweep_main

        sweep_main(args.config)

    elif args.command == "evaluate":
        from .evaluate import main as eval_main

        eval_main(args.config, args.base, args.adapter, args.tag)

    elif args.command == "judge":
        from .judge import main as judge_main

        judge_main(args.config)

    elif args.command == "forgetting":
        from .forgetting import main as forget_main

        forget_main(args.config, args.base, args.adapter)

    elif args.command == "serve":
        from .serve import serve

        serve(args.base, args.adapter, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
