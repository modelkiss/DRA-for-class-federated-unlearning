"""CLI for stage 7: diffusion guided generation."""
from __future__ import annotations

import argparse
import json

from src.attacks.data_reconstruction import AdaptiveGenerationConfig
from src.workflows import DiffusionGenerationStageConfig, run_diffusion_generation_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 7: Diffusion guided generation")
    parser.add_argument("--training-dir", default="outputs/stages/diffusion_training")
    parser.add_argument("--inference-dir", default="outputs/stages/label_inference")
    parser.add_argument("--forgetting-summary", required=True)
    parser.add_argument("--sfi-summary", default=None)
    parser.add_argument("--output", default="outputs/stages/diffusion_generation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--images-per-batch", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--target-min", type=float, default=None)
    parser.add_argument("--target-max", type=float, default=None)
    parser.add_argument("--accuracy-margin", type=float, default=None)
    parser.add_argument("--save-rejected", action="store_true")
    return parser.parse_args()


def build_adaptive_config(args: argparse.Namespace) -> AdaptiveGenerationConfig | None:
    overrides = [
        args.images_per_batch,
        args.max_batches,
        args.guidance_scale,
        args.inference_steps,
        args.target_min,
        args.target_max,
        args.accuracy_margin,
    ]
    if not any(value is not None for value in overrides) and not args.save_rejected:
        return None
    adaptive = AdaptiveGenerationConfig()
    if args.images_per_batch is not None:
        adaptive.images_per_batch = args.images_per_batch
    if args.max_batches is not None:
        adaptive.max_batches = args.max_batches
    if args.guidance_scale is not None:
        adaptive.guidance_scale = args.guidance_scale
    if args.inference_steps is not None:
        adaptive.num_inference_steps = args.inference_steps
    if args.target_min is not None:
        adaptive.target_accuracy_min = args.target_min
    if args.target_max is not None:
        adaptive.target_accuracy_max = args.target_max
    if args.accuracy_margin is not None:
        adaptive.accuracy_margin = args.accuracy_margin
    if args.save_rejected:
        adaptive.save_rejected = True
    return adaptive


def main() -> None:
    args = parse_args()
    adaptive = build_adaptive_config(args)
    config = DiffusionGenerationStageConfig(
        training_dir=args.training_dir,
        inference_dir=args.inference_dir,
        forgetting_summary=args.forgetting_summary,
        sensitive_summary=args.sfi_summary,
        output_dir=args.output,
        device=args.device,
        adaptive=adaptive,
    )
    result = run_diffusion_generation_stage(config)
    print(json.dumps({"summary_path": str(result.summary_path)}, indent=2, ensure_ascii=False))
    print("已完成阶段7：扩散模型引导生成完成。")


if __name__ == "__main__":
    main()
