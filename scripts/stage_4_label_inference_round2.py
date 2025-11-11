"""CLI for stage 4: second round label inference."""
from __future__ import annotations

import argparse
import json

from src.workflows import LabelInferenceStageConfig, run_label_inference_stage_two


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: Label inference round 2")
    parser.add_argument("--inference-dir", default="outputs/stages/label_inference")
    parser.add_argument("--forgetting-summary", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabelInferenceStageConfig(
        dataset=args.dataset,
        forgetting_summary=args.forgetting_summary,
        output_dir=args.inference_dir,
        device=args.device,
    )
    result = run_label_inference_stage_two(config)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print("已完成阶段4：确定最可能的遗忘标签。")


if __name__ == "__main__":
    main()
