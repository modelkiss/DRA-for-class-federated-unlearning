"""CLI for stage 6: diffusion model fine-tuning."""
from __future__ import annotations

import argparse
import json

from src.attacks.data_reconstruction import (
    AdaptiveGenerationConfig,
    BaseDiffusionTrainingConfig,
    DataReconstructionConfig,
    SensitiveFeatureFinetuneConfig,
)
from src.workflows import DiffusionTrainingStageConfig, run_diffusion_training_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6: Diffusion fine-tuning")
    parser.add_argument("--forgetting-summary", required=True)
    parser.add_argument("--inference-dir", default="outputs/stages/label_inference")
    parser.add_argument("--sfi-summary", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default="outputs/stages/diffusion_training")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-dir", default="models/diffusion/base")
    parser.add_argument("--controlnet-dir", default="models/diffusion/controlnet")
    parser.add_argument("--base-steps", type=int, default=1500)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--base-batch", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--method", default="lora")
    parser.add_argument("--sensitive-steps", type=int, default=800)
    parser.add_argument("--sensitive-lr", type=float, default=5e-5)
    parser.add_argument("--disable-sensitive", action="store_true")
    parser.add_argument("--images-per-batch", type=int, default=1024)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--target-min", type=float, default=0.0)
    parser.add_argument("--target-max", type=float, default=0.6)
    parser.add_argument("--accuracy-margin", type=float, default=0.05)
    parser.add_argument("--save-rejected", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recon_config = DataReconstructionConfig(
        base=BaseDiffusionTrainingConfig(
            model_id=args.model_dir,
            max_train_steps=args.base_steps,
            learning_rate=args.base_lr,
            batch_size=args.base_batch,
            resolution=args.resolution,
            lora_rank=args.lora_rank,
            method=args.method,
        ),
        sensitive=SensitiveFeatureFinetuneConfig(
            enabled=not args.disable_sensitive,
            controlnet_model_id=args.controlnet_dir,
            max_train_steps=args.sensitive_steps,
            learning_rate=args.sensitive_lr,
        ),
        adaptive=AdaptiveGenerationConfig(
            images_per_batch=args.images_per_batch,
            max_batches=args.max_batches,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.inference_steps,
            target_accuracy_min=args.target_min,
            target_accuracy_max=args.target_max,
            accuracy_margin=args.accuracy_margin,
            save_rejected=args.save_rejected,
        ),
    )

    config = DiffusionTrainingStageConfig(
        dataset=args.dataset,
        forgetting_summary=args.forgetting_summary,
        inference_dir=args.inference_dir,
        sensitive_summary=args.sfi_summary,
        output_dir=args.output,
        device=args.device,
        config=recon_config,
    )
    result = run_diffusion_training_stage(config)
    print(json.dumps({"summary_path": str(result.summary_path)}, indent=2, ensure_ascii=False))
    print("已完成阶段6：扩散模型初步训练完成。")


if __name__ == "__main__":
    main()
