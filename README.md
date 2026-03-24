<div align="center">

<img src="assets/logo.svg" width="75%">

### A Simple Baseline for Streaming Video Understanding

<p>
<a href="https://arxiv.org/abs/2604.02317"><img src="https://img.shields.io/badge/arXiv-2604.02317-b31b1b.svg" alt="Paper"></a>
<a href="https://huggingface.co/papers/2604.02317"><img src="https://img.shields.io/badge/🤗-Paper_Page-yellow" alt="HF Paper"></a>
<a href="https://simple-stream.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
</p>

<img src="assets/teaser.png" width="90%">

A sliding-window baseline that feeds only the most recent ***N*** frames to an off-the-shelf VLM **matches or surpasses** published streaming video understanding models. No memory bank, no retrieval, no compression.

</div>

## 🚀 News

📄 **[2026/04]** SimpleStream paper is released.

💻 **[2026/04]** Code and evaluation scripts are open-sourced.

## ✨ Highlights

- **Simple yet strong.** With only 4 recent frames, SimpleStream reaches **67.7%** on OVO-Bench and **80.59%** on StreamingBench, surpassing all published streaming methods.
- **Perception-memory trade-off.** Adding historical context improves recall but consistently degrades real-time perception, which dominates aggregate scores.
- **Training-free.** SimpleStream uses off-the-shelf VLMs (Qwen2.5-VL, Qwen3-VL) with zero fine-tuning.

## 📊 Main Results

<p align="center">
<img src="assets/main_results.png" width="90%">
</p>

## 🛠️ Getting Started

### Environment Setup

```bash
conda create -n simplestream python=3.10 -y
conda activate simplestream
pip install -r requirements.txt
# Optional: faster attention backend
pip install flash-attn --no-build-isolation
```

### Models

Downloaded automatically from HuggingFace on first run:
- `Qwen/Qwen3-VL-8B-Instruct` (primary)
- `Qwen/Qwen2.5-VL-7B-Instruct` (cross-validation)

### Data Preparation

- **OVO-Bench**: Download from [OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench). Place annotations at `data/ovo_bench/ovo_bench_new.json` and chunked videos at `data/ovo_bench/chunked_videos/`.
- **StreamingBench**: Download from [StreamingBench](https://github.com/Infini-AI-Lab/StreamingBench). Place questions at `data/streamingbench/questions_real.json` and videos at `data/streamingbench/videos/`.

## 🧪 Experiments

<details>
<summary><b>Qwen3-VL on OVO-Bench</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 \
    main_experiments/eval_qwen3vl_ovo.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir main_experiments/results/ovo_qwen3vl_recent8 \
    --recent_frames_only 8 \
    --chunk_duration 1.0 \
    --fps 1.0
```

Or use the convenience launcher for 4-GPU:
```bash
bash main_experiments/run_qwen3vl_ovo_4gpu.sh
```
</details>

<details>
<summary><b>Qwen2.5-VL on OVO-Bench</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 \
    main_experiments/eval_qwen25vl_ovo.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --anno_path data/ovo_bench/ovo_bench_new.json \
    --chunked_dir data/ovo_bench/chunked_videos \
    --result_dir main_experiments/results/ovo_qwen25vl_recent8 \
    --recent_frames_only 8 \
    --chunk_duration 1.0 \
    --fps 1.0
```
</details>

<details>
<summary><b>StreamingBench</b></summary>

`--top-k 0` disables retrieval and keeps only the most recent chunks.

```bash
CUDA_VISIBLE_DEVICES=0 python main_experiments/eval_streamingbench.py \
    --anno-path data/streamingbench/questions_real.json \
    --video-dir data/streamingbench/videos \
    --top-k 0 \
    --recent-frames-only 4 \
    --chunk-duration 1.0 \
    --fps 1.0 \
    --output-dir main_experiments/results/streamingbench_recent4
```
</details>

<details>
<summary><b>Efficiency Benchmark</b></summary>

Measures TTFT, throughput, and memory from a user-provided source video.

```bash
CUDA_VISIBLE_DEVICES=0 python efficiency/eval_efficiency.py \
    --source-video /path/to/your/source_video.mp4 \
    --model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --chunk-size 8 \
    --recent-frames 4
```
</details>

<details>
<summary><b>Scoring</b></summary>

```bash
python scoring/score_ovo_bench.py \
    --result_path main_experiments/results/ovo_qwen3vl_recent8/qwen3vl_results_*.json
```
</details>

## 📢 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{simplestream2026,
  title={A Simple Baseline for Streaming Video Understanding},
  author={Shen, Yujiao and Tian, Shulin and Yang, Jingkang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2604.02317},
  year={2026}
}
```

## 🙏 Acknowledgement

- [Qwen-VL](https://github.com/QwenLM/Qwen2.5-VL): the VLM backbone used in our experiments.
- [OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench) and [StreamingBench](https://github.com/Infini-AI-Lab/StreamingBench): the evaluation benchmarks.
