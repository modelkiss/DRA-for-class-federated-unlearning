"""CLI for the stage 1 federated model training workflow."""
from __future__ import annotations

import argparse
import json

from src.workflows import TrainingStageConfig, run_model_training_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Federated model training")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--iid", action="store_true", help="Use IID partitioning (default non-IID Dirichlet)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--client-lr", type=float, default=0.01)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--target-accuracy", type=float, default=0.8, help="Required per-class accuracy threshold")
    parser.add_argument("--device", default=None, help="Torch device identifier")
    parser.add_argument("--output", default="outputs/stages/training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingStageConfig(
        dataset=args.dataset,
        num_clients=args.num_clients,
        iid=args.iid,
        dirichlet_alpha=args.dirichlet_alpha,
        client_learning_rate=args.client_lr,
        local_epochs=args.local_epochs,
        client_batch_size=args.batch_size,
        client_fraction=args.client_fraction,
        max_rounds=args.max_rounds,
        target_class_accuracy=args.target_accuracy,
        device=args.device,
        output_dir=args.output,
    )
    result = run_model_training_stage(config)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print(f"已完成阶段1：共执行 {result.aggregator_steps} 次聚合。")


if __name__ == "__main__":
    main()
