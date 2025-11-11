"""CLI for stage 3: first round label inference."""
from __future__ import annotations

import argparse
import json

from src.workflows import LabelInferenceStageConfig, run_label_inference_stage_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Label inference round 1")
    parser.add_argument("--forgetting-summary", required=True, help="Path to stage 2 forgetting summary JSON")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default="outputs/stages/label_inference")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabelInferenceStageConfig(
        dataset=args.dataset,
        forgetting_summary=args.forgetting_summary,
        output_dir=args.output,
        device=args.device,
    )
    result = run_label_inference_stage_one(config)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print("已完成阶段3：生成候选标签并计算准确率。")


if __name__ == "__main__":
    main()
