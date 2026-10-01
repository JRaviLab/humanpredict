"""Build data/human_labeled_biosamples.csv from data/bacteria_BioSample.csv.

Mirrors the labeling logic in `data exploration/data_prep.ipynb`:
    is_human =  1  host_name matches a human term
    is_human = -1  host_name present, not human
    is_human =  0  host_name missing (unlabeled pool)

Each group is deduplicated on isolation_comments, as in the notebook.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "bacteria_BioSample.csv"
OUT_PATH = ROOT / "data" / "human_labeled_biosamples.csv"

HUMAN_NAMES = ['homo', 'homo sapiens', 'human', 'patient', 'homo sapien', 'man', 'woman', '9606']

NONHUMAN_NAMES = [
    "Homo sapiens, freshwater",
    "homo sapiens, soil",
    "HUMAN/FOOD",
    "non human primate",
    "Human/Culture",
    "Homo sapiens and Environment",
    "non-human primate",
    "Non-human primate",
    "In human and animal",
    "dog and human",
    "Non-human",
    "non-human",
    "marmot and human",
    "mouse/human",
    "human and other animals",
    "Human, livestock",
    "Non Human Primate",
    "Sheep and human",
    "Homo sapiens and other mammals",
]


def build(df: pd.DataFrame) -> pd.DataFrame:
    # Longer phrases first, wrapped in word boundaries
    pattern = r'\b(?:' + '|'.join(re.escape(n) for n in sorted(HUMAN_NAMES, key=len, reverse=True)) + r')\b'
    mask = df['host_name'].str.contains(pattern, case=False, na=False, regex=True)

    human_host = df[mask]
    human_host = human_host[~human_host['host_name'].isin(NONHUMAN_NAMES)]

    non_human_host = df.dropna(subset='host_name')
    non_human_host = non_human_host[~non_human_host['biosample_accession'].isin(human_host['biosample_accession'])]

    # Unlabeled: samples with NO host_name (previously dropna, which selected labeled rows)
    nans = df[df['host_name'].isna()]

    human_host = human_host.drop_duplicates(subset='isolation_comments').assign(is_human=1)
    non_human_host = non_human_host.drop_duplicates(subset='isolation_comments').assign(is_human=-1)
    nans = nans.drop_duplicates(subset='isolation_comments').assign(is_human=0)

    return pd.concat([human_host, non_human_host, nans], ignore_index=True)


def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)
    final_df = build(df)

    labeled = final_df.loc[final_df['is_human'] != 0, 'biosample_accession']
    unlabeled = final_df.loc[final_df['is_human'] == 0]
    assert final_df['biosample_accession'].is_unique, "duplicate accessions in output"
    assert not unlabeled['biosample_accession'].isin(labeled).any(), "unlabeled rows overlap labeled rows"
    assert unlabeled['host_name'].isna().all(), "unlabeled rows must have no host_name"

    final_df.to_csv(OUT_PATH, index=False)
    print(final_df['is_human'].value_counts().to_string())
    print(f"Wrote {len(final_df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
