"""CLI for stage 5: sensitive feature inference."""
from __future__ import annotations

import argparse
import json

from src.workflows import SensitiveFeatureStageConfig, run_sensitive_feature_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5: Sensitive feature inference")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--inference-dir", default="outputs/stages/label_inference")
    parser.add_argument("--output", default="outputs/stages/sensitive_features")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-classes", type=int, default=1)
    parser.add_argument("--mask-quantile", type=float, default=0.8)
    parser.add_argument("--mask-min", type=float, default=0.2)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-patches", type=int, default=8)
    parser.add_argument("--dct-components", type=int, default=32)
    parser.add_argument("--num-prototypes", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SensitiveFeatureStageConfig(
        dataset=args.dataset,
        inference_dir=args.inference_dir,
        output_dir=args.output,
        device=args.device,
        max_classes=args.max_classes,
        mask_quantile=args.mask_quantile,
        mask_min_threshold=args.mask_min,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        dct_components=args.dct_components,
        num_prototypes=args.num_prototypes,
    )
    result = run_sensitive_feature_stage(config)
    print(json.dumps({"summary_path": str(result.summary_path)}, indent=2, ensure_ascii=False))
    print("已完成阶段5：敏感特征推理结果已保存。")


if __name__ == "__main__":
    main()
