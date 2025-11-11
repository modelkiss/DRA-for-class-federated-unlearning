"""CLI for stage 2: federated model forgetting."""
from __future__ import annotations

import argparse
import json

from src.defenses.differential_privacy import DifferentialPrivacyConfig
from src.federated.aggregation import AggregationConfig
from src.workflows import ForgettingStageConfig, run_model_forgetting_stage


def _parse_json_dict(raw: str | None, *, argument: str) -> dict:
    if raw is None:
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI validation
        raise argparse.ArgumentTypeError(f"参数 --{argument} 必须是合法的 JSON 对象: {exc}") from exc
    if not isinstance(value, dict):
        example = '{"key": "value"}'
        raise argparse.ArgumentTypeError(
            f"参数 --{argument} 需要提供 JSON 对象，例如 {example}"
        )
    return value


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
    parser.add_argument(
        "--aggregation-method",
        default=None,
        help="Override aggregation mechanism (默认使用训练阶段配置)",
    )
    parser.add_argument(
        "--aggregation-params",
        default=None,
        help="Aggregation mechanism parameters in JSON format",
    )
    parser.add_argument(
        "--dp-method",
        default=None,
        help="Override differential privacy method (默认使用训练阶段配置)",
    )
    parser.add_argument(
        "--dp-params",
        default=None,
        help="Differential privacy parameters in JSON format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregation_cfg = None
    if args.aggregation_method is not None:
        aggregation_cfg = AggregationConfig(
            mechanism=args.aggregation_method,
            parameters=_parse_json_dict(args.aggregation_params, argument="aggregation-params"),
        )
    dp_cfg = None
    if args.dp_method is not None:
        dp_cfg = DifferentialPrivacyConfig(
            method=args.dp_method,
            parameters=_parse_json_dict(args.dp_params, argument="dp-params"),
        )
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
        aggregation=aggregation_cfg,
        dp_config=dp_cfg,
    )
    result = run_model_forgetting_stage(config)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print("已完成阶段2：模型遗忘完成。")


if __name__ == "__main__":
    main()
