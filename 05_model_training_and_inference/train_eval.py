#!/usr/bin/env python3
"""
IFCB Plankton Models: Baselines + Simple Domain Adaptation

CLI subcommands:
  train  -> train on a source domain (optionally with CORAL); optional BN adaptation on target
  eval   -> evaluate a trained checkpoint on any dataset root

Expected folder layout for datasets (symlinked or real):
  <DATA_ROOT>/
    class_A/*.png|jpg|tif
    class_B/*.png|jpg|tif
    ...

Typical runs
------------
# 1) Source-only training on SYKE, evaluate on Tångesund (later via eval)
python train_eval.py train \
  --source-root \
    /cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/03_dataset_creation/exports/dataset_symlinked/SYKE_plankton_IFCB_Utö_2021 \
  --target-root \
    /cfs/klemming/projects/supr/snic2020-6-126/projects/amime/DDLS_course_project/03_dataset_creation/exports/dataset_symlinked/smhi_ifcb_tångesund \
  --run-dir runs/syke_to_tangesund/source_only \
  --method source_only --epochs 12

# 2) CORAL training (source CE + coral loss on source/target features)
python train_eval.py train \
  --source-root .../SYKE_plankton_IFCB_Utö_2021 \
  --target-root .../smhi_ifcb_svea_baltic_proper \
  --run-dir runs/syke_to_baltic/coral \
  --method coral --coral-lambda 0.5 --epochs 12

# 3) BN adaptation (recalibrate BN stats on target after source-only training)
python train_eval.py train \
  --source-root .../SYKE_plankton_IFCB_Utö_2021 \
  --target-root .../smhi_ifcb_svea_skagerrak_kattegat \
  --run-dir runs/syke_to_skagerrak/bn_adapt \
  --method bn_adapt --bn-passes 1 --epochs 12

# 4) Evaluate a checkpoint on any root
python train_eval.py eval \
  --ckpt runs/syke_to_tangesund/source_only/best.ckpt \
  --data-root .../smhi_ifcb_tångesund \
  --report-out reports/eval_syke_to_tangesund.json

Notes
-----
* Images are normalized with ImageNet mean/std and resized to 224.
* If your IFCB images are grayscale, we auto-convert to 3-ch by repeating channels.
* You can exclude specific classes (e.g., Unclassifiable) with --exclude-classes.
* Uses macro-F1 as primary score, plus accuracy.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

# Try to use sklearn for metrics, but fall back to local implementation
try:
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


# --------------------- Utils ---------------------

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: str | Path):
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# --------------------- Data ---------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_transforms(train: bool, strong_aug: bool = False):
    aug = []
    if train:
        # Light, geometry-safe augs for microscopy-like images
        if strong_aug:
            aug += [transforms.TrivialAugmentWide()]
        else:
            aug += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        aug += [transforms.RandomResizedCrop(224, scale=(0.8, 1.0))]
    else:
        aug += [transforms.Resize(256), transforms.CenterCrop(224)]

    aug += [
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    # We inject a grayscale->RGB adapter in the dataset wrapper (below) so transforms assume 3ch float tensor
    return transforms.Compose([transforms.ToTensor()] + aug)


class ImageFolderFiltered(datasets.ImageFolder):
    """ImageFolder that (optionally) excludes certain class names and handles grayscale images."""
    def __init__(self, root: str | Path, transform=None, exclude_classes: Optional[Iterable[str]] = None):
        super().__init__(root=str(root), transform=transform)

        # Map class name -> index
        name_to_idx = {cls_name: i for cls_name, i in self.class_to_idx.items()}
        if exclude_classes:
            exclude = set(exclude_classes)
            keep_idx = set(i for c, i in self.class_to_idx.items() if c not in exclude)
            # filter samples
            self.samples = [s for s in self.samples if s[1] in keep_idx]
            self.targets = [s[1] for s in self.samples]
            # Rebuild class_to_idx compactly for kept classes
            kept_classes = sorted([c for c in self.classes if c not in exclude])
            self.class_to_idx = {c: j for j, c in enumerate(kept_classes)}
            idx_remap = {name_to_idx[c]: j for j, c in enumerate(kept_classes)}
            self.targets = [idx_remap[t] for t in self.targets]
            self.samples = [(p, idx_remap[t]) for (p, _), t in zip(self.samples, self.targets)]
            self.classes = kept_classes

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        img = self.loader(path)  # PIL
        # Ensure 3 channels: IFCB is often grayscale; repeat channels if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def make_loaders(
    source_root: str | Path,
    target_root: Optional[str | Path] = None,
    batch_size: int = 64,
    workers: int = 4,
    strong_aug: bool = False,
    exclude: Optional[List[str]] = None,
):
    tr_tf = make_transforms(train=True, strong_aug=strong_aug)
    te_tf = make_transforms(train=False)

    # --- source dataset
    src_base = ImageFolderFiltered(source_root, transform=tr_tf, exclude_classes=exclude)
    # class maps are defined by the *source* set (after optional exclusion)
    class_to_idx = src_base.class_to_idx
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # split train/val on source
    val_ratio = 0.1 if len(src_base) >= 1000 else 0.2
    n_val = max(1, int(len(src_base) * val_ratio))
    n_train = len(src_base) - n_val
    src_train_ds, src_val_ds = torch.utils.data.random_split(
        src_base, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # no aug for val
    src_val_ds.dataset.transform = te_tf

    src_train_loader = DataLoader(src_train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    src_val_loader   = DataLoader(src_val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # --- target loader (optional)
    tgt_loader = None
    if target_root is not None:
        tgt_ds = ImageFolderFiltered(target_root, transform=te_tf, exclude_classes=exclude)
        _align_target_to_source(tgt_ds, class_to_idx)
        if len(tgt_ds.samples) == 0:
            print("⚠️  Warning: no overlapping classes between source and target after alignment; skipping target loader.")
        else:
            tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return src_train_loader, src_val_loader, tgt_loader, class_to_idx, idx_to_class



# --------------------- Model & Losses ---------------------

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone: str = 'resnet18', pretrained: bool = True):
        super().__init__()
        if backbone == 'resnet18':
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
            self.classifier = nn.Linear(feat_dim, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x, return_features: bool = False):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits


class CoralLoss(nn.Module):
    """CORAL loss: distance between source & target covariance matrices.
    Reference: https://arxiv.org/abs/1607.01719
    """
    def forward(self, src_feats: torch.Tensor, tgt_feats: torch.Tensor) -> torch.Tensor:
        # src_feats, tgt_feats: [B, D]
        def _cov(f):
            f = f - f.mean(dim=0, keepdim=True)
            # unbiased=False (population cov) for stability
            c = (f.T @ f) / (f.shape[0] - 1)
            return c
        Cs = _cov(src_feats)
        Ct = _cov(tgt_feats)
        return F.mse_loss(Cs, Ct)


# --------------------- Training / Eval ---------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> Dict:
    if _HAVE_SK:
        # Always evaluate over the full label set [0..C-1]
        label_indices = list(range(len(labels)))
        macro_f1 = float(f1_score(y_true, y_pred, average='macro', labels=label_indices))
        acc = float((y_true == y_pred).mean())
        rep = classification_report(
            y_true, y_pred,
            labels=label_indices,
            target_names=labels,
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=label_indices).tolist()
        return {"macro_f1": macro_f1, "accuracy": acc, "report": rep, "confusion_matrix": cm}
    # Fallback minimal metrics unchanged...
    acc = float((y_true == y_pred).mean())
    n_classes = len(labels)
    f1s = []
    for k in range(n_classes):
        tp = np.sum((y_true == k) & (y_pred == k))
        fp = np.sum((y_true != k) & (y_pred == k))
        fn = np.sum((y_true == k) & (y_pred != k))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    return {"macro_f1": macro_f1, "accuracy": acc}



def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            ys.append(y.numpy())
            yh.append(preds)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(yh)

    # Try to pull class names from the underlying dataset if available
    classes = None
    ds = loader.dataset
    if hasattr(ds, "classes"):
        classes = ds.classes
    elif hasattr(ds, "dataset") and hasattr(ds.dataset, "classes"):
        classes = ds.dataset.classes
    else:
        # fallback: numeric labels
        n_classes = int(y_true.max()) + 1 if y_true.size else 0
        classes = [str(i) for i in range(n_classes)]

    return compute_metrics(y_true, y_pred, classes)



def bn_adaptation(model: nn.Module, target_loader: DataLoader, device: torch.device, passes: int = 1):
    """Recompute BN running stats on target data without changing weights.
    Set model to train() so BN updates, but keep grads off.
    """
    if target_loader is None:
        return
    model.train()
    with torch.no_grad():
        for _ in range(passes):
            for x, _ in target_loader:
                x = x.to(device, non_blocking=True)
                _ = model(x)


def train(
    source_root: str,
    target_root: Optional[str],
    run_dir: str,
    method: str = 'source_only',
    backbone: str = 'resnet18',
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    workers: int = 4,
    strong_aug: bool = False,
    exclude_classes: Optional[List[str]] = None,
    coral_lambda: float = 0.5,
    bn_passes: int = 1,
    seed: int = 1337,
):
    assert method in {'source_only', 'coral', 'bn_adapt'}
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_train, src_val, tgt_eval, class_to_idx, idx_to_class = make_loaders(
        source_root, target_root, batch_size, workers, strong_aug, exclude_classes
    )

    ensure_dir(run_dir)
    save_json({"class_to_idx": class_to_idx}, Path(run_dir) / 'class_to_idx.json')

    model = ResNetClassifier(num_classes=len(class_to_idx), backbone=backbone, pretrained=True).to(device)

    # Class weights to mitigate imbalance (inverse frequency on source train split)
    counts = np.bincount([y for _, y in src_train.dataset], minlength=len(class_to_idx))
    # Use the Subset indices to avoid loading images
    train_subset = src_train.dataset  # torch.utils.data.Subset
    base_ds = train_subset.dataset    # ImageFolderFiltered (has .targets)
    idxs = train_subset.indices       # indices into base_ds
    labels = [base_ds.targets[i] for i in idxs]
    counts = np.bincount(labels, minlength=len(class_to_idx))
    inv = 1.0 / np.clip(counts, 1, None)
    weights = inv / inv.sum() * len(inv)
    ce_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
    coral_loss = CoralLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = -1.0
    best_path = Path(run_dir) / 'best.ckpt'

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        # For CORAL, we need an iterator over target batches (no labels needed)
        tgt_iter = iter(tgt_eval) if (method == 'coral' and tgt_eval is not None) else None

        for xb, yb in src_train:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, feats_s = model(xb, return_features=True)
                loss = ce_loss(logits, yb)

                if method == 'coral' and tgt_iter is not None:
                    try:
                        xt, _ = next(tgt_iter)
                    except StopIteration:
                        tgt_iter = iter(tgt_eval)
                        xt, _ = next(tgt_iter)
                    xt = xt.to(device, non_blocking=True)
                    _, feats_t = model(xt, return_features=True)
                    loss = loss + coral_lambda * coral_loss(feats_s, feats_t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        # Source-val metrics
        val_metrics = evaluate(model, src_val, device)
        mean_loss = total_loss / max(1, n_batches)

        # Save last + best
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': {
                'method': method,
                'backbone': backbone,
                'epochs': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'weight_decay': weight_decay,
                'strong_aug': strong_aug,
                'exclude_classes': exclude_classes,
                'coral_lambda': coral_lambda,
                'bn_passes': bn_passes,
            },
        }, Path(run_dir) / 'last.ckpt')

        if val_metrics['macro_f1'] > best_val:
            best_val = val_metrics['macro_f1']
            torch.save({'model_state': model.state_dict()}, best_path)

        print(f"Epoch {epoch:03d} | loss {mean_loss:.4f} | src-val macroF1 {val_metrics['macro_f1']:.4f} acc {val_metrics['accuracy']:.4f}")

    # Optional BN adaptation stage
    if method == 'bn_adapt' and tgt_eval is not None:
        # Load best pre-adaptation weights
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state['model_state'])
        bn_adaptation(model, tgt_eval, device, passes=bn_passes)
        torch.save({'model_state': model.state_dict()}, Path(run_dir) / 'best_bn_adapted.ckpt')

    # Final evaluation on target if provided
    results = {
        'source_val': evaluate(model, src_val, device),
    }
    if tgt_eval is not None:
        results['target_eval'] = evaluate(model, tgt_eval, device)

    save_json(results, Path(run_dir) / 'results.json')
    print("Saved:", best_path, Path(run_dir) / 'results.json')


def load_model_for_eval(ckpt_path: str | Path, num_classes: int, backbone: str = 'resnet18') -> nn.Module:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetClassifier(num_classes=num_classes, backbone=backbone, pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    # ckpt may be {model_state: ...} or full trainer dict
    state_dict = state['model_state'] if 'model_state' in state else state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def eval_only(ckpt: str, data_root: str, exclude_classes: Optional[List[str]] = None, report_out: Optional[str] = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = ImageFolderFiltered(data_root, transform=make_transforms(train=False), exclude_classes=exclude_classes)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    model = load_model_for_eval(ckpt, num_classes=len(ds.classes))
    metrics = evaluate(model, loader, device)
    if report_out:
        save_json(metrics, report_out)
        print("Wrote:", report_out)
    print(json.dumps({k: v for k, v in metrics.items() if k in ('macro_f1', 'accuracy')}, indent=2))


# --------------------- CLI ---------------------

def build_parser():
    p = argparse.ArgumentParser(description="IFCB plankton baselines + domain adaptation")
    sub = p.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='Train a model')
    t.add_argument('--source-root', required=True)
    t.add_argument('--target-root', default=None, help='Optional target root (needed for CORAL or BN adapt & target eval)')
    t.add_argument('--run-dir', required=True)
    t.add_argument('--method', choices=['source_only', 'coral', 'bn_adapt'], default='source_only')
    t.add_argument('--backbone', default='resnet18')
    t.add_argument('--epochs', type=int, default=12)
    t.add_argument('--batch-size', type=int, default=64)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--weight-decay', type=float, default=1e-4)
    t.add_argument('--workers', type=int, default=4)
    t.add_argument('--strong-aug', action='store_true')
    t.add_argument('--exclude-classes', nargs='*', default=None, help='e.g. Unclassifiable')
    t.add_argument('--coral-lambda', type=float, default=0.5)
    t.add_argument('--bn-passes', type=int, default=1)
    t.add_argument('--seed', type=int, default=1337)

    e = sub.add_parser('eval', help='Evaluate a checkpoint on a dataset root')
    e.add_argument('--ckpt', required=True)
    e.add_argument('--data-root', required=True)
    e.add_argument('--exclude-classes', nargs='*', default=None)
    e.add_argument('--report-out', default=None)

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == 'train':
        train(
            source_root=args.source_root,
            target_root=args.target_root,
            run_dir=args.run_dir,
            method=args.method,
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            workers=args.workers,
            strong_aug=args.strong_aug,
            exclude_classes=args.exclude_classes,
            coral_lambda=args.coral_lambda,
            bn_passes=args.bn_passes,
            seed=args.seed,
        )
    elif args.cmd == 'eval':
        eval_only(
            ckpt=args.ckpt,
            data_root=args.data_root,
            exclude_classes=args.exclude_classes,
            report_out=args.report_out,
        )


def _align_target_to_source(tgt_ds, src_class_to_idx):
    # Remap target samples to source indices (by class name) and drop non-overlapping classes
    name_by_idx = tgt_ds.classes
    new_samples = []
    for p, y in tgt_ds.samples:
        cls = name_by_idx[y]
        if cls in src_class_to_idx:
            new_samples.append((p, src_class_to_idx[cls]))
    tgt_ds.samples = new_samples
    tgt_ds.targets = [y for _, y in new_samples]
    tgt_ds.class_to_idx = src_class_to_idx.copy()
    # rebuild classes in the *source* order
    tgt_ds.classes = [c for c, i in sorted(src_class_to_idx.items(), key=lambda kv: kv[1])]


if __name__ == '__main__':
    main()
