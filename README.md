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
- **Federated Training** – configurable FedAvg or FedProx orchestration with
  differential privacy (DP-SGD, LDP-FL, Adaptive DP-FL, RDP-FL) and secure
  aggregation (SecAgg, AHSecAgg, Pairwise Masking/FastSecAgg) that can be used
  jointly.
**Forgetting** – remove a selected class from every client and apply
  FedEraser calibration, adaptive FedAF optimisation, or one-shot classifier
  surgery so that three distinct forgetting mechanisms can be compared.
- **Attacks** – infer the forgotten label using per-class accuracy, confusion,
  and gradient-difference signals, and reconstruct representative samples with
  a text-to-image diffusion generator.
- **Reporting** – store reconstructed tensors, serialized models, and metadata
  summarizing attack success under different defense regimes.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision diffusers

# 1. Federated model training (自动迭代直至每类准确率 ≥ 80%)
python scripts/stage_1_train_model.py --dataset cifar10 --num-clients 10 --iid \
    --target-accuracy 0.8 --max-rounds 50 --output outputs/stages/training

# 2. 模型遗忘（默认 one-shot）
python scripts/stage_2_model_forgetting.py --training-summary \
    outputs/stages/training/training_summary.json --target-class 6 \
    --method oneshot --output outputs/stages/forgetting

# 3. 标签推理第一轮（计算测试集准确率+推选出60%候选标签）
python scripts/stage_3_label_inference_round1.py --forgetting-summary \
    outputs/stages/forgetting/forgetting_summary.json

# 4. 标签推理第二轮（确认最高得分标签）
python scripts/stage_4_label_inference_round2.py

# 5. 敏感特征推理
python scripts/stage_5_sensitive_feature_inference.py --dataset cifar10

# 6. 扩散模型初步训练（完全离线，需提前放置模型权重）
python scripts/stage_6_diffusion_training.py --forgetting-summary \
    outputs/stages/forgetting/forgetting_summary.json --sfi-summary \
    outputs/stages/sensitive_features/sensitive_feature_summary.json

# 7. 扩散模型引导生成
python scripts/stage_7_diffusion_generation.py --forgetting-summary \
    outputs/stages/forgetting/forgetting_summary.json
```

每个阶段都会在 `outputs/stages/<stage_name>/` 下写入 JSON 摘要文件，后续阶段只需
读取前一阶段的输出即可继续执行，保证全流程可控可追踪。若需覆盖参数，可使用各
阶段脚本提供的命令行选项；所有配置都会记录在对应的 summary 中。

### 离线资源准备

为确保训练与推理全过程无需联网，仓库预先创建了下列目录用于存放所需的模型权重
与缓存，请将对应的文件手动放置进去：

```
models/diffusion/base/        # Stable Diffusion 或自有底模
models/diffusion/controlnet/  # ControlNet 相关权重（可选）
models/diffusion/cache/       # 其他辅助缓存
```

若不需要敏感特征引导，可在阶段 6/7 中加上 `--disable-sensitive` 或不提供
ControlNet 目录。所有脚本均默认读取本地目录，不会触发任何在线下载。

## Project structure

```
src/
  attacks/                Label inference and reconstruction attacks
  data/                   Dataset loading and federated partitioning
  federated/              Aggregation (FedAvg/FedProx) with DP + secure aggregation
  forgetting/             Class removal and post-processing routines
  models/                 Neural network architectures for each dataset
  utils/                  Shared utilities (logging, metrics)
scripts/
  stage_1_train_model.py             Stage 1 federated training CLI
  stage_2_model_forgetting.py        Stage 2 forgetting CLI
  stage_3_label_inference_round1.py  Stage 3 label inference (round 1)
  stage_4_label_inference_round2.py  Stage 4 label inference (round 2)
  stage_5_sensitive_feature_inference.py
                                     Stage 5 sensitive feature inference
  stage_6_diffusion_training.py      Stage 6 diffusion fine-tuning
  stage_7_diffusion_generation.py    Stage 7 diffusion guided generation
  export_reconstructions.py          Decode `reconstructed.pt` tensors into images
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

1. 按顺序完成阶段 1–7，确保 `outputs/stages/diffusion_generation/` 中生成
   `reconstruction_summary.json` 与导出的图像批次。
2. 在同一目录或自定义目录下执行上述命令，脚本会自动反归一化张量并
   输出单张图片与（若指定 `--grid`）拼图网格。
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
- Extend diffusion-based attacks by wrapping alternative text-to-image
  pipelines inside `DiffusionReconstructor`.
- Add alternative forgetting strategies inside
  `src/forgetting/class_forgetting.py`.

## License

This project is provided as-is for research prototyping.