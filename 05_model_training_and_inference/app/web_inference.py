# app/web_inference.py

import io
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# ---- Make sure we can import train_eval from the parent directory ----
import sys
THIS_DIR = Path(__file__).resolve().parent                # .../05_model_training_and_inference/app
PROJECT_ROOT = THIS_DIR.parent                            # .../05_model_training_and_inference
sys.path.append(str(PROJECT_ROOT))

from train_eval import ResNetClassifier, IMAGENET_MEAN, IMAGENET_STD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Paths to your two best runs ----
MODEL_CONFIGS = {
    "syke_to_baltic_bn_adapt": {
        "ckpt": PROJECT_ROOT / "runs/syke_to_baltic/bn_adapt/best_bn_adapted.ckpt",
        "class_to_idx": PROJECT_ROOT / "runs/syke_to_baltic/bn_adapt/class_to_idx.json",
    },
    "tangesund_to_skagerrak_source_only": {
        "ckpt": PROJECT_ROOT / "runs/tangesund_to_skagerrak/source_only/best.ckpt",
        "class_to_idx": PROJECT_ROOT / "runs/tangesund_to_skagerrak/source_only/class_to_idx.json",
    },
}

# ---- Same eval transform as in train_eval.make_transforms(train=False) ----
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

MODELS = {}
IDX_TO_CLASS = {}


def _load_class_mapping(path: Path):
    with open(path) as f:
        data = json.load(f)
    # file was saved as {"class_to_idx": {...}}
    if "class_to_idx" in data:
        class_to_idx = data["class_to_idx"]
    else:
        class_to_idx = data
    # keys are class names, values are indices
    idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def _load_model(name: str):
    if name in MODELS:
        return MODELS[name], IDX_TO_CLASS[name]

    cfg = MODEL_CONFIGS[name]
    class_to_idx, idx_to_class = _load_class_mapping(cfg["class_to_idx"])

    model = ResNetClassifier(num_classes=len(class_to_idx), backbone="resnet18", pretrained=False)
    state = torch.load(cfg["ckpt"], map_location=DEVICE)
    state_dict = state["model_state"] if "model_state" in state else state
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    MODELS[name] = model
    IDX_TO_CLASS[name] = idx_to_class
    return model, idx_to_class


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)  # [1, C, H, W]
    return x.to(DEVICE)


@torch.no_grad()
def predict_all(image_bytes: bytes, topk: int = 3):
    x = _preprocess(image_bytes)
    results = {}

    for name in MODEL_CONFIGS.keys():
        model, idx_to_class = _load_model(name)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k=topk)

        preds = []
        for p, idx in zip(top_probs.cpu().tolist(), top_idxs.cpu().tolist()):
            cls_name = idx_to_class.get(int(idx), f"class_{idx}")
            preds.append((cls_name, float(p)))

        results[name] = preds

    return results
