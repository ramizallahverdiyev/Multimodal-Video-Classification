from __future__ import annotations

import argparse
import yaml
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, List
from collections import Counter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from src.data_pipeline.loaders.dataset_classes.tubesense_dataset import TubeSenseDataset, tubesense_collate  # noqa: E402
from src.data_pipeline.loaders.dataset_classes.folder_multimodal_dataset import FolderMultimodalDataset, folder_collate  # noqa: E402
from src.models.architectures.multimodal_model import TubeSenseModel  # noqa: E402


def load_config(path: Path) -> Dict:
    with path.open() as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: TubeSenseModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="train"):
        images = batch["image"].to(device)
        audio = batch["audio"].to(device)
        text = {k: v.to(device) for k, v in batch["text"].items()}
        labels = batch["label"].to(device)

        valid_mask = labels >= 0
        if not torch.any(valid_mask):
            continue

        optimizer.zero_grad()
        logits = model(images, audio, text)
        loss = criterion(logits[valid_mask], labels[valid_mask])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def save_checkpoint(model: TubeSenseModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[+] Saved checkpoint to {path}")


def evaluate(model: TubeSenseModel, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            audio = batch["audio"].to(device)
            text = {k: v.to(device) for k, v in batch["text"].items()}
            labels = batch["label"].to(device)
            logits = model(images, audio, text)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    acc = correct / total if total else 0.0
    avg_loss = running_loss / max(len(loader), 1)
    return acc, avg_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TubeSense multimodal model.")
    parser.add_argument("--config", type=Path, default=Path("config/model_config.yml"), help="Config YAML path")
    parser.add_argument("--data-csv", type=Path, default=Path("data/processed/metadata.csv"), help="Metadata CSV path")
    parser.add_argument(
        "--dataset-type",
        choices=["metadata", "folder"],
        default="metadata",
        help="Use CSV-based dataset or folder-based (frames/audio/text by label folder).",
    )
    parser.add_argument("--frames-root", type=Path, default=Path("data/interim/frames"), help="Frames root for folder dataset")
    parser.add_argument("--audio-root", type=Path, default=Path("data/interim/audio"), help="Audio root for folder dataset")
    parser.add_argument("--text-root", type=Path, default=Path("data/interim/text"), help="Text root for folder dataset")
    parser.add_argument("--split-file", type=Path, help="Optional split CSV (label,movie_id,split) for folder dataset")
    parser.add_argument("--split", type=str, default="train", help="Which split to load from split-file")
    parser.add_argument("--val-split", type=str, default="val", help="Validation split name in split-file")
    parser.add_argument("--use-class-weights", action="store_true", help="Use class weights to rebalance loss")
    parser.add_argument("--freeze-text", action="store_true", help="Freeze text encoder during training")
    parser.add_argument("--freeze-vision", action="store_true", help="Freeze vision encoder during training")
    parser.add_argument("--freeze-audio", action="store_true", help="Freeze audio encoder during training")
    parser.add_argument("--weighted-sampler", action="store_true", help="Use WeightedRandomSampler for train split")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--max-epochs", type=int, help="Override max epochs")
    parser.add_argument("--resume", type=Path, help="Optional checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    class_names = [c.lower() for c in cfg["class_names"]]
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def make_loader(split_name: str, shuffle: bool = True, use_sampler: bool = False) -> DataLoader:
        sampler = None
        dataset = None
        if args.dataset_type == "metadata":
            dataset = TubeSenseDataset(
                csv_path=args.data_csv,
                label_to_idx=label_to_idx,
                tokenizer_name=cfg["text"]["pretrained_model"],
                max_length=64,
                sample_rate=cfg["audio"]["sample_rate"],
                segment_seconds=cfg["audio"].get("segment_seconds", 20),
            )
            if use_sampler and args.weighted_sampler:
                counts = Counter(dataset.df["label"].str.lower())
                weights_per_class = {lbl: len(dataset.df) / (cnt * len(counts)) for lbl, cnt in counts.items()}
                sample_weights = [weights_per_class.get(str(lbl).lower(), 1.0) for lbl in dataset.df["label"]]
                sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        else:
            frames_root = args.frames_root
            audio_root = args.audio_root
            text_root = args.text_root if args.text_root.exists() else None
            dataset = FolderMultimodalDataset(
                frames_root=frames_root,
                audio_root=audio_root,
                text_root=text_root,
                label_to_idx=label_to_idx,
                split_file=args.split_file,
                split=split_name,
                tokenizer_name=cfg["text"]["pretrained_model"],
                max_length=64,
                sample_rate=cfg["audio"]["sample_rate"],
                segment_seconds=cfg["audio"].get("segment_seconds", 20),
            )
            if use_sampler and args.weighted_sampler:
                label_counts = Counter([item[0] for item in dataset.items])
                total = sum(label_counts.values())
                weights_per_class = {lbl: total / (cnt * len(label_counts)) for lbl, cnt in label_counts.items()}
                sample_weights = [weights_per_class[item[0]] for item in dataset.items]
                sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(
            dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=cfg["training"]["num_workers"],
            collate_fn=tubesense_collate if args.dataset_type == "metadata" else folder_collate,
        )

    train_loader = make_loader(args.split, shuffle=True, use_sampler=True)
    val_loader: Optional[DataLoader] = None
    if args.split_file and args.val_split:
        val_loader = make_loader(args.val_split, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TubeSenseModel(
        num_classes=len(class_names),
        text_model=cfg["text"]["pretrained_model"],
    ).to(device)

    if args.freeze_text:
        for p in model.text.parameters():
            p.requires_grad = False
    if args.freeze_vision:
        for p in model.vision.parameters():
            p.requires_grad = False
    if args.freeze_audio:
        for p in model.audio.parameters():
            p.requires_grad = False

    lr = float(args.learning_rate) if args.learning_rate else float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state)
        print(f"[+] Resumed weights from {args.resume}")

    def compute_class_weights(loader: DataLoader, num_classes: int) -> Optional[torch.Tensor]:
        if not args.use_class_weights:
            return None
        counts = Counter()
        for lbl in getattr(loader.dataset, "items", []):
            # FolderMultimodalDataset items are (label, video_id)
            if isinstance(lbl, tuple):
                label_idx = label_to_idx.get(lbl[0], None)
            else:
                label_idx = None
            if label_idx is not None:
                counts[label_idx] += 1
        if not counts:
            return None
        total = sum(counts.values())
        weights: List[float] = []
        for cls_idx in range(num_classes):
            freq = counts.get(cls_idx, 1)
            weights.append(total / (freq * num_classes))
        return torch.tensor(weights, dtype=torch.float32, device=device)

    class_weights = compute_class_weights(train_loader, len(class_names))
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = -1.0
    max_epochs = int(args.max_epochs) if args.max_epochs else int(cfg["training"]["max_epochs"])
    for epoch in range(max_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        log = f"[epoch {epoch}] train_loss={loss:.4f}"
        if val_loader:
            val_acc, val_loss = evaluate(model, val_loader, device)
            log += f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / f"{cfg['experiment_name']}_best.pt"
                save_checkpoint(model, ckpt_path)
        print(log)
        ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / f"{cfg['experiment_name']}_epoch{epoch}.pt"
        save_checkpoint(model, ckpt_path)


if __name__ == "__main__":
    main()
