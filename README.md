# Federated Class Unlearning Attack Benchmark

This repository implements an end-to-end experimental harness for studying
class-level unlearning requests in federated learning systems. The pipeline
closely follows the specification in the prompt and supports the following key
components:

- **Datasets** – CIFAR-10/100, MNIST and Fashion-MNIST with IID or Dirichlet
  non-IID partitions across clients.
- **Models** – lightweight CNNs for CIFAR (VGG-style) and MNIST (LeNet) along
  with a factory that can be extended to ResNet-style models.
- **Federated Training** – standard FedAvg orchestration with optional
  differential privacy (Gaussian noise + clipping) and secure aggregation.
- **Forgetting** – remove a selected class from every client and continue
  training or invoke an auxiliary logit-suppression procedure.
- **Attacks** – infer the forgotten label using per-class accuracy deltas and
  reconstruct representative samples by optimizing noise to reactivate the
  removed class.
- **Reporting** – store reconstructed tensors, serialized models, and metadata
  summarizing attack success under different defense regimes.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision
python scripts/run_pipeline.py --dataset cifar10 --num-clients 10 --rounds 5 \
    --forget-rounds 3 --target-class 6 --reconstructions 4 --iid
```

Outputs are stored in the `outputs/` directory (configurable through
`--output`). Besides the reconstructed tensors and attack report, the pipeline
logs baseline and post-forgetting accuracies together with the inferred class.

## Project structure

```
src/
  attacks/                Label inference and reconstruction attacks
  data/                   Dataset loading and federated partitioning
  federated/              FedAvg implementation with DP and secure aggregation
  forgetting/             Class removal and post-processing routines
  models/                 Neural network architectures for each dataset
  utils/                  Shared utilities (logging, metrics)
scripts/
  run_pipeline.py         Command-line interface for the full experiment
```

The design keeps the modules composable so that new datasets, models, or attack
strategies can be plugged in without touching the orchestrator.

## Extending

- Register additional models in `src/models/nets.py` and expose them via
  `build_model`.
- Implement new reconstruction methods by subclassing or replacing
  `GradientReconstructor`.
- Add alternative forgetting strategies inside
  `src/forgetting/class_forgetting.py`.

## License

This project is provided as-is for research prototyping.
