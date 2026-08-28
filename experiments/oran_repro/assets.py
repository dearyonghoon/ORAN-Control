from pathlib import Path
import json
import torch
from .config import holdout_tag
from .data import load_normalization
from .model import load_model

def holdout_dir(exp11_root, slicing_id):
    return Path(exp11_root) / holdout_tag(slicing_id)

def load_holdout_assets(exp11_root, slicing_id, device):
    d = holdout_dir(exp11_root, slicing_id)
    norm = load_normalization(d / "normalization.npz")
    model = load_model(d / "flow_numeric.pt", device)
    with open(d / "selected_guidance.json", "r", encoding="utf-8") as f:
        guidance = json.load(f)
    return d, model, norm, guidance

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
