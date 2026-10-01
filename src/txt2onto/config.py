from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    for key in ("processed_dir", "splits_dir", "embeddings_dir", "artifacts_dir", "reports_dir"):
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    return cfg
