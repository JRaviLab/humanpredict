"""PubMedBERT sentence embeddings, cached by text hash (PLAN.md section 2)."""
import numpy as np
import pandas as pd

EMB_FILE = "emb.npy"
INDEX_FILE = "emb_index.parquet"


def load_model(emb_cfg: dict):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(emb_cfg["model"], revision=emb_cfg["revision"], device=emb_cfg["device"])
    if emb_cfg["fp16"] and emb_cfg["device"].startswith("cuda"):
        model.half()
    model.max_seq_length = emb_cfg["max_seq_length"]
    return model


def encode(texts: list[str], emb_cfg: dict, model=None) -> np.ndarray:
    model = model or load_model(emb_cfg)
    emb = model.encode(
        texts,
        batch_size=emb_cfg["batch_size"],
        normalize_embeddings=emb_cfg["normalize"],
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return emb.astype(np.float32)  # fp16 inference returns float16


def load_cache(emb_dir) -> tuple[np.ndarray, pd.Series]:
    """Return (embeddings [N, 768] float32 memmap, Series text_hash -> row)."""
    if not (emb_dir / EMB_FILE).exists():
        return np.empty((0, 0), dtype=np.float32), pd.Series(dtype="int64")
    emb = np.load(emb_dir / EMB_FILE, mmap_mode="r")
    index = pd.read_parquet(emb_dir / INDEX_FILE)
    return emb, pd.Series(index["row"].values, index=index["text_hash"].values)


def update_cache(texts: pd.DataFrame, cfg: dict) -> dict:
    """Embed any text_hash in `texts` (columns text_hash, text) not already cached."""
    emb_dir = cfg["paths"]["embeddings_dir"]
    old_emb, old_index = load_cache(emb_dir)
    todo = texts.drop_duplicates("text_hash")
    todo = todo[~todo["text_hash"].isin(old_index.index)]
    if todo.empty:
        return {"embedded": 0, "cached": len(old_index)}

    new_emb = encode(todo["text"].tolist(), cfg["embedding"])
    emb = np.vstack([np.asarray(old_emb), new_emb]) if len(old_index) else new_emb
    index = pd.DataFrame({
        "text_hash": np.concatenate([old_index.index.values, todo["text_hash"].values]),
        "row": np.arange(len(emb)),
    })
    np.save(emb_dir / EMB_FILE, emb)
    index.to_parquet(emb_dir / INDEX_FILE, index=False)
    return {"embedded": len(todo), "cached": len(index)}


def lookup(hashes, emb: np.ndarray, index: pd.Series) -> np.ndarray:
    """Embeddings for a sequence of text hashes, in order."""
    return np.asarray(emb[index.loc[list(hashes)].values])
