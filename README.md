# TubeSense: Multimodal Video Classification

TubeSense downloads videos (e.g., trailers) and fuses vision, audio, and text to predict content genres (action, comedy, etc.). This repo is a lean, end-to-end playground: data sourcing, preprocessing, and training with a fusion model.

## Quick Start
1) Env: Python 3.10+, FFmpeg installed (`ffmpeg -version`).  
2) Install: `python -m venv .venv && .venv\Scripts\activate && pip install -r config/requirements.txt`  
3) Build split (uses existing frames/audio/text):  
   `python tools/make_split.py --root data/interim --out data/processed/split.csv`  
4) Train (folder dataset):  
```
.venv\Scripts\python src/training/scripts/train_model.py ^
  --dataset-type folder ^
  --split-file data/processed/split.csv ^
  --split train --val-split val ^
  --config config/model_config.yml ^
  --use-class-weights --weighted-sampler
```
Checkpoints are written to `results/models/`; best model is `tubesense_fusion_best.pt`.

## Data Pipeline (summary)
- **Sourcing:** `src/data_pipeline/sourcing` for YouTube trailers and IMDB-derived labels.
- **Preprocessing:** `extract_from_raw.py` creates frames (JPEG), audio (WAV), and ASR text; outputs live under `data/interim/`.
- **Dataset:** `FolderMultimodalDataset` reads frames/audio/text per label folder; splits from `data/processed/split.csv`.
- **Model:** `src/models/architectures` combines vision/audio/text encoders with `fusion_layer.py`.
- **Training:** `src/training/scripts/train_model.py` supports class weights, weighted sampler, and validation tracking.
