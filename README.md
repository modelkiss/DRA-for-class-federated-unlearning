# DRA-for-class-federated-unlearning
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
training, perform logit suppression, or reinitialise the classifier head for
  the target class so that three distinct forgetting mechanisms can be
  compared.
- **Attacks** – infer the forgotten label using per-class accuracy, confusion,
  and gradient-difference signals, and reconstruct representative samples via
  gradient inversion or an optional diffusion generator.
- **Reporting** – store reconstructed tensors, serialized models, and metadata
  summarizing attack success under different defense regimes.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision
python scripts/run_pipeline.py --dataset cifar10 --num-clients 10 --rounds 5 \
   --forget-rounds 3 --target-class 6 --reconstructions 4 --iid \
    --forgetting-method classifier_reinit --reconstruction-method diffusion \
    --diffusion-model-id runwayml/stable-diffusion-v1-5 --secure-aggregation bonawitz2017 \
    --dp-sigma 0.8 --dp-mechanism laplace
```

Outputs are stored in the `outputs/` directory (configurable through
`--output`). Besides the reconstructed tensors and attack report, the pipeline
logs baseline/post-forgetting accuracies and inferred classes, and now exports
three heatmaps (`heatmap_accuracy_delta_<dataset>.png`、
`heatmap_confusion_delta_<dataset>.png`、`heatmap_gradient_delta_<dataset>.png`)
that visualise the per-class accuracy 变化、混淆矩阵差值以及梯度范数差异。默认使用
Matplotlib 的 `coolwarm` 色图，可通过 `--heatmap-cmap` 修改；若不希望生成热力图，
可追加 `--no-heatmaps`。

常用命令行开关说明：

- `--forgetting-method`：在 `fine_tune`、`logit_suppression`、`classifier_reinit`
  三种遗忘策略之间切换。
- `--reconstruction-method`：选择 `gradient` 或 `diffusion` 重建方式。若使用
  扩散模型，可配合 `--diffusion-model-id`、`--diffusion-guidance`、
  `--diffusion-steps` 与 `--diffusion-negative-prompt` 微调生成质量。
- `--secure-aggregation`：模拟不同的安全聚合协议
  （`bonawitz2017`、`shamir`、`homomorphic`）。
- `--dp-sigma`、`--dp-mechanism` 与 `--dp-clip`：控制差分隐私噪声强度与分布
  （Gaussian/Laplace/Student-t），用于评估攻击在不同隐私级别下的鲁棒性。
- `--gradient-batches` 与 `--gradient-params`：指定梯度差分统计所用的数据
  批次数以及关注的参数子串，从而影响生成的梯度热力图。

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
  export_reconstructions.py
                          Utility to decode `reconstructed.pt` tensors into images
```

The design keeps the modules composable so that new datasets, models, or attack
strategies can be plugged in without touching the orchestrator.

## Exporting reconstructed samples

After running the main pipeline you can convert the stored tensor file into
standard image formats. The exporter automatically attempts to read the
dataset name from `metrics.json` or `inference.json`, so in most cases you only
need to specify the output directory:

```bash
python scripts/export_reconstructions.py \
    --reconstructions outputs/reconstructed.pt \
    --output outputs/reconstructed_images --grid
```

快速操作步骤：

1. 先运行 `scripts/run_pipeline.py`，确保 `outputs/` 目录中生成了
   `reconstructed.pt`、`metrics.json` 以及 `inference.json`。
2. 在同一目录下执行上述命令，脚本会自动反归一化张量并输出单张图片
   与（若指定 `--grid`）拼图网格。
3. 输出文件夹中默认包含 `class_<id>_000.png` 等文件名；若同时保存了
   ground-truth 信息，则前缀会写成 `pred6_gt2` 以示区分。

参数说明：

- 若自动探测失败，可通过 `--dataset` 显式指定（可选值：`cifar10`、`cifar100`、
  `mnist`、`fashionmnist`）。
- `--inference` 与 `--metadata` 默认指向 `outputs/` 目录下的 JSON，若路径不同需显式设置。
- `--start-index` 可设定导出图片的起始编号；`--format` 支持 `png`、`jpg`、`jpeg`。
- `--grid-columns` 控制拼图时每行图片数，适合与 `--grid` 搭配使用。

The script automatically inverts the dataset normalisation and writes
individual images such as `class_6_000.png`. If the inference metadata is
provided, predicted and ground-truth class identifiers are reflected in the
filenames. Because the reconstructions are generated approximations, they
cannot be matched one-to-one with the exact forgotten training samples.

## Extending

- Register additional models in `src/models/nets.py` and expose them via
  `build_model`.
- Implement new reconstruction methods by subclassing or replacing
  `GradientReconstructor`.
- Add alternative forgetting strategies inside
  `src/forgetting/class_forgetting.py`.

## License

This project is provided as-is for research prototyping.