"""CLI for stage 2: federated model forgetting."""
from __future__ import annotations

import argparse
import json

from src.workflows import ForgettingStageConfig, run_model_forgetting_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: Federated model forgetting")
    parser.add_argument("--training-summary", required=True, help="Path to stage 1 training summary JSON")
    parser.add_argument("--target-class", type=int, required=True)
    parser.add_argument("--dataset", default=None, help="Dataset name override")
    parser.add_argument("--method", default="oneshot", choices=["oneshot", "fed_eraser", "fedaf"])
    parser.add_argument("--output", default="outputs/stages/forgetting")
    parser.add_argument("--device", default=None)
    parser.add_argument("--client-lr", type=float, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--client-fraction", type=float, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--fedaf-rounds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ForgettingStageConfig(
        dataset=args.dataset,
        target_class=args.target_class,
        training_summary=args.training_summary,
        method=args.method,
        output_dir=args.output,
        client_learning_rate=args.client_lr,
        local_epochs=args.local_epochs,
        client_batch_size=args.batch_size,
        client_fraction=args.client_fraction,
        num_clients=args.num_clients,
        device=args.device,
        fedaf_rounds=args.fedaf_rounds,
    )
    result = run_model_forgetting_stage(config)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print("已完成阶段2：模型遗忘完成。")


if __name__ == "__main__":
    main()
