"""Convert reconstructed tensors into image files for inspection."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Tuple

import torch
from torchvision.utils import save_image

LOGGER = logging.getLogger(__name__)

# Normalization statistics mirror the preprocessing used during training.
NORMALIZATION_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "mnist": ((0.1307,), (0.3081,)),
    "fashionmnist": ((0.1307,), (0.3081,)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reconstructed tensors saved by run_pipeline.py to image files.",
    )
    parser.add_argument(
        "--reconstructions",
        type=Path,
        default=Path("outputs/reconstructed.pt"),
        help="Path to the tensor file produced by run_pipeline.py.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(NORMALIZATION_STATS.keys()),
        default=None,
        help=(
            "Dataset name used for the experiment. If omitted the script attempts to read it "
            "from --metadata or --inference files."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reconstructed_images"),
        help="Directory where decoded images will be written.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Image file format for the exported samples.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix; defaults to the predicted/ground-truth class when available.",
    )
    parser.add_argument(
        "--inference",
        type=Path,
        default=Path("outputs/inference.json"),
        help=(
            "Optional path to inference.json for annotating filenames with class IDs. "
            "Defaults to outputs/inference.json."
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs/metrics.json"),
        help=(
            "Optional path to metrics.json (or any JSON containing a 'dataset' field). "
            "Used to auto-detect the dataset when --dataset is not specified."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for numbering the exported images.",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Also export a single grid image containing all reconstructions.",
    )
    parser.add_argument(
        "--grid-columns",
        type=int,
        default=4,
        help="Number of images per row when saving a grid (only if --grid is set).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (e.g. INFO, DEBUG).",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="[%(levelname)s] %(message)s")


def load_reconstructions(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Reconstruction file not found: {path}")

    data = torch.load(path, map_location="cpu")
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, (list, tuple)):
        tensor = torch.stack([_ensure_tensor(item) for item in data], dim=0)
    elif isinstance(data, dict) and "samples" in data:
        tensor = _ensure_tensor(data["samples"])
    else:
        raise TypeError(
            "Unsupported reconstruction container. Expected a Tensor, sequence, or dict with a 'samples' key."
        )

    if tensor.dim() != 4:
        raise ValueError(f"Expected reconstructions with shape (N, C, H, W) but got {tuple(tensor.shape)}")
    return tensor.float()


def _ensure_tensor(value: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.cpu()
    return torch.tensor(value, dtype=torch.float32)


def denormalize(tensor: torch.Tensor, stats: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> torch.Tensor:
    mean, std = stats
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(1, -1, 1, 1)
    return tensor * std_tensor + mean_tensor


def determine_prefix(args: argparse.Namespace, metadata: dict | None) -> str:
    if args.prefix:
        return args.prefix
    if metadata is None:
        return args.reconstructions.stem

    predicted = metadata.get("predicted_class")
    ground_truth = metadata.get("ground_truth")
    if predicted is None:
        return args.reconstructions.stem

    if ground_truth is None:
        return f"class_{predicted}"
    if int(predicted) == int(ground_truth):
        return f"class_{predicted}"
    return f"pred{predicted}_gt{ground_truth}"


def _load_optional_json(path: Path | None, *, strict: bool = False) -> dict | None:
    if path is None:
        return None

    if not path.exists():
        if strict:
            raise FileNotFoundError(f"Metadata not found: {path}")
        LOGGER.debug("Metadata file %s not found; skipping", path)
        return None

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dataset(args: argparse.Namespace, *metadata_sources: dict | None) -> str:
    if args.dataset is not None:
        return args.dataset

    for source in metadata_sources:
        if isinstance(source, dict):
            dataset = source.get("dataset")
            if dataset in NORMALIZATION_STATS:
                LOGGER.info("Detected dataset '%s' from metadata", dataset)
                return dataset

    valid = ", ".join(sorted(NORMALIZATION_STATS))
    raise ValueError(
        "Unable to determine dataset. Provide --dataset explicitly or supply metadata containing "
        f"a 'dataset' field (valid options: {valid})."
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    inference_metadata = _load_optional_json(args.inference)
    metrics_metadata = _load_optional_json(args.metadata)
    dataset = resolve_dataset(args, metrics_metadata, inference_metadata)

    LOGGER.info("Loading reconstructions from %s", args.reconstructions)
    reconstructions = load_reconstructions(args.reconstructions)
    LOGGER.info(
        "Loaded %d samples with shape CxHxW=%s", reconstructions.shape[0], reconstructions.shape[1:]
    )

    stats = NORMALIZATION_STATS[dataset]
    images = denormalize(reconstructions, stats).clamp(0.0, 1.0)

    args.output.mkdir(parents=True, exist_ok=True)
    prefix = determine_prefix(args, inference_metadata)
    if inference_metadata is None:
        LOGGER.info(
            "No inference metadata supplied; using '%s' as the filename prefix.",
            prefix,
        )
    else:
        predicted = inference_metadata.get("predicted_class")
        ground_truth = inference_metadata.get("ground_truth")
        if predicted is None:
            LOGGER.info("Inference metadata provided without a predicted class; using prefix '%s'", prefix)
        elif ground_truth is None or int(predicted) == int(ground_truth):
            LOGGER.info("Using predicted class %s as the filename prefix.", predicted)
        else:
            LOGGER.info(
                "Predicted class %s differs from recorded ground-truth %s; prefix '%s' encodes both.",
                predicted,
                ground_truth,
                prefix,
            )

    LOGGER.info("Saving images to %s with prefix '%s'", args.output, prefix)
    for idx, image in enumerate(images, start=args.start_index):
        destination = args.output / f"{prefix}_{idx:03d}.{args.format}"
        save_image(image, destination)

    if args.grid:
        grid_path = args.output / f"{prefix}_grid.{args.format}"
        save_image(images, grid_path, nrow=args.grid_columns)
        LOGGER.info("Saved grid image to %s", grid_path)

    LOGGER.info(
        "Export complete. Note: reconstructions are generated approximations and cannot be matched one-to-one "
        "with the exact forgotten training samples."
    )


if __name__ == "__main__":
    main()