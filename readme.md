# Few-Shot Synthetic Image Attribution: Identifying Unseen Generators with Limited Samples

<p align="center">
  <a href="https://arxiv.org/abs/2509.25682"><img src="https://img.shields.io/badge/arXiv-paper-red?style=flat&logo=arXiv" alt="Paper" height="25"></a>
  <a href="https://huggingface.co/datasets/MoeNew/OmniFake"><img src="https://img.shields.io/badge/Dataset-OmniFake-blue?style=flat&logo=huggingface&logoColor=white" alt="Dataset" height="25"></a>
  <a href="https://huggingface.co/MoeNew/OmniDFA"><img src="https://img.shields.io/badge/Model_Weights-OmniDFA-orange?style=flat&logo=huggingface&logoColor=white" alt="Model Weights" height="25"></a>
</p>

## 📰 News

* **[2026.06.27]** 🔥 We are uploading **OmniFake** to HuggingFace. Due to network bandwidth limitations, the full dataset will be available within 1 week.
* **[2026.06.26]** ✅ We have released the model weights for **OmniDFA**. Check out **[** [HuggingFace](https://huggingface.co/MoeNew/OmniDFA) **]**.
* **[2026.06.24]** ✅ We have open-sourced the **OmniDFA** codebase. Check out our **[** [Paper](https://arxiv.org/abs/2509.25682) **]**.

Generative models now forge photorealistic images that defy visual scrutiny, collapsing the boundary between authentic and synthetic. Existing attribution methods operate in a **closed-set** manner — they require retraining whenever a new generator appears, making them quickly obsolete in the face of rapidly evolving generation technology.

We introduce **few-shot attribution**: a new paradigm that identifies unseen generators using only a handful of reference samples, with no retraining required. To support this, we construct **OmniFake**, a large-scale synthetic image dataset containing **1.17 million images** from **45 distinct generators**, designed to ensure pairwise class separability for attribution research.

We further propose **OmniDFA** (Omni Detector and Few-shot Attributor), a unified framework that simultaneously handles two tasks:

- **Authenticity Detection**: distinguishes real images from AI-generated ones via a one-class hypersphere formulation, achieving state-of-the-art generalization on unseen generators.
- **Few-Shot Source Attribution**: given K examples from N generator classes, identifies the source of a query image via prototype-based classification.

OmniDFA also demonstrates robust zero-shot detection on the [GenImage](https://github.com/GenImage-Dataset/GenImage) and [Chameleon](https://drive.google.com/file/d/1QLYJMhy0CbBVT01BLkkw7KPPL5BpmxnH) ([AIDE](https://github.com/shilinyan99/AIDE)) benchmarks.


## Dataset: OmniFake

OmniFake covers 45 AI image generators plus real images, organized into three cross-validation parts and a zero-shot split for out-of-distribution evaluation.


| Split | Train generators | Eval generators |
|-------|-----------------|-----------------|
| Part 1 | Parts 2 + 3 (30 generators) | Part 1 (15 generators) |
| Part 2 | Parts 1 + 3 (30 generators) | Part 2 (15 generators) |
| Part 3 | Parts 1 + 2 (30 generators) | Part 3 (15 generators) |
| Zero-shot | 33 generators | Open-source benchmarks |

More detailed class sources are listed in Appendix Table 1 of the paper.

### Download

Download the OmniFake dataset from the following link: [HuggingFace](https://huggingface.co/datasets/MoeNew/OmniFake)

After downloading, extract the archives using the provided helper script:

```bash
# Usage: ./scripts/data/unpack_all.sh <target_dir> [source_dir]
bash scripts/data/unpack_all.sh /share/project/OmniFake /path/to/downloaded/zips
```

The extracted directory should follow this structure:

```
OmniFake/
├── train/
│   ├── real/
│   │   └── *.jpg
│   ├── ADM/
│   │   └── *.jpg
│   ├── FLUX_Dev/
│   │   └── *.jpg
│   ├── DALLE_3/
│   │   └── *.jpg
│   ├── Midjourney_V6/
│   │   └── *.jpg
│   └── ...                 # 45 generators total
└── val/
    ├── real/
    ├── ADM/
    ├── FLUX_Dev/
    └── ...
```


## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Our experiments are conducted with PyTorch 2.6.0 and CUDA 12.4.

### 2. Download Model Weights

Download pretrained model weights from: [HuggingFace](https://huggingface.co/your-org/OmniDFA) | [BaiduYun](https://pan.baidu.com/s/1OxiASGLzQLByYxkGRxKzdg) (code: `eccv`)

### 3. Training

Edit `scripts/train.sh` to set `DATA_PATH` to your OmniFake root directory, then run:

```bash
bash scripts/train.sh
```

The script uses `torchrun` with 8 GPUs by default. Three training configurations are provided (one per cross-validation fold); uncomment the desired command to switch folds:

```bash
# Train on parts 1 & 2 → eval on part 3
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --use_bf16 --total_epochs 20 \
  --data_path /share/project/OmniFake \
  --fake_file_path datasets/split/part3/trainlist.txt \
  --real_file_path datasets/split/part3/reallist.txt

```

### 4. Authenticity Detection (Real vs. Fake)

Edit `scripts/eval_detection.sh` to set `DATA_PATH` and `CKPT_PATH`, then run:

```bash
bash scripts/eval_detection.sh
```

To evaluate on [GenImage](https://github.com/GenImage-Dataset/GenImage) and [Chameleon](https://github.com/shilinyan99/AIDE) benchmarks, first download the respective datasets, then replace `--fake_file_path` with the corresponding class list under `datasets/split/zero-shot/`:

```bash
# GenImage OOD evaluation
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 eval_detection.py \
  --use_bf16 --data_path /path/to/genimage \
  --fake_file_path datasets/split/zero-shot/vallist_genimage.txt \
  --real_file_path datasets/split/zero-shot/reallist_genimage.txt \
  --ckpt_path your_ckpt_path

# Chameleon OOD evaluation
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 eval_detection.py \
  --use_bf16 --data_path /path/to/chameleon \
  --fake_file_path datasets/split/zero-shot/vallist_chamelon.txt \
  --real_file_path datasets/split/zero-shot/reallist_chamelon.txt \
  --ckpt_path your_ckpt_path
```

### 5. Few-Shot Source Attribution

Edit `scripts/eval_attribution.sh` to set `DATA_PATH`, `CKPT_PATH`, `N_WAY`, and `K_SHOT`, then run:

```bash
bash scripts/eval_attribution.sh
```

Default configuration: **5-way 10-shot**, 5 queries per class, 10 000 episodes. Supports `--n_way 5` or `--n_way 15`.

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=4 eval_attribution.py \
  --use_bf16 --n_way 5 --k_shot 10 \
  --data_path /share/project/OmniFake \
  --fake_file_path datasets/split/part3/vallist.txt \
  --ckpt_path your_ckpt_path
```



## Acknowledgements

We thank the teams behind [GenImage](https://github.com/GenImage-Dataset/GenImage), [WildFake](https://github.com/Whut-YiRong/WildFake), [Fake Image Dataset](https://huggingface.co/datasets/InfImagine/FakeImageDataset), and other open-source AIGI datasets whose data partially contributed to OmniFake. We also thank the open-source generative model authors on HuggingFace (see Appendix for the full list) for providing state-of-the-art image generators that enabled the construction of our dataset.


## Citation

If you find OmniFake or OmniDFA useful in your research, please cite:

```bibtex
@article{omnidfa2026,
  title={OmniDFA: Omni AI-Generated Image Detector with Few-Shot Attribution},
  author={Shiyu Wu and Shuyan Li and Jing Li and Jing Liu and Yequan Wang},
  journal={arXiv preprint arXiv:2509.25682},
  year={2026}
}
```
