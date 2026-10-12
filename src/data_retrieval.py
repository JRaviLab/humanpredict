import json
import os
import shutil
import tempfile

import duckdb
import pandas as pd

BIOSAMPLE_URL = "https://data-omicidx.cancerdatasci.org/biosample/parquet/biosamples.parquet"
OUTPUT_PATH = "data/Whole_BioSample.parquet"

# ---------------------------------------------------------------------------
# 1. Load the mapping table: raw attribute name (key) -> manual label (manual_map)
# ---------------------------------------------------------------------------
key_map = pd.read_csv("data/key_map.csv", index_col=0)
# Match case-insensitively; strip stray outer whitespace so keys line up with the data
key_map["key"] = key_map["key"].str.lower().str.strip()
key_map = key_map.drop_duplicates(subset="key")

# {label: [raw keys...]}, e.g. {'host_name': ['host', 'host taxid', ...], ...}
label_to_keys = key_map.groupby("manual_map")["key"].apply(list).to_dict()

# "comments" is not in the mapping table; it is a simple catch-all bin for a few note-like fields
label_to_keys["comments"] = ["field_notes", "note", "comment", "sample comment", "notes"]

metadata_cols = list(label_to_keys)
all_keys = [k for keys in label_to_keys.values() for k in keys]


def sql_list(values):
    """Render a Python list as a SQL string list, escaping single quotes (some keys contain them, e.g. 'host')."""
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def label_expr(label, keys):
    """
    One output column per label: keep only attributes whose name is in this label's key list,
    take their values, and join them with " | ". If a sample has none, the empty string becomes NULL.
    """
    return f"""
        NULLIF(array_to_string(list_transform(
            list_filter(attribute_recs, x -> lower(trim(x.attribute_name)) IN ({sql_list(keys)})),
            y -> y.value
        ), ' | '), '') AS {label}"""


# ---------------------------------------------------------------------------
# 2. Build and run the query (one generated column per label)
# ---------------------------------------------------------------------------
label_columns = ",".join(label_expr(label, keys) for label, keys in label_to_keys.items())

query = f"""
    SELECT
        accession AS biosample_accession,

        -- THE SAMPLE ORGANISM (Standardized Top-Level Columns)
        taxon_id AS sample_taxon_id,
        taxonomy_name AS sample_taxonomy_name,

        -- Mapped metadata labels
        {label_columns},

        -- Which mapped raw keys this sample actually had (used only for the retrieval check below)
        list_filter(
            list_transform(attribute_recs, x -> lower(trim(x.attribute_name))),
            n -> n IN ({sql_list(all_keys)})
        ) AS matched_keys

    FROM read_parquet('{BIOSAMPLE_URL}')
    WHERE attribute_recs IS NOT NULL
"""

# Memory: do all heavy work inside DuckDB, backed by a temporary on-disk database so it can
# spill to disk instead of exhausting RAM. Only the final bacterial rows are loaded into pandas.
tmp_dir = tempfile.mkdtemp(prefix="biosample_duckdb_")
con = duckdb.connect(os.path.join(tmp_dir, "work.duckdb"))
con.execute("SET memory_limit = '12GB';")
con.execute(f"SET temp_directory = '{tmp_dir}';")
con.execute("SET preserve_insertion_order = false;")  # lets DuckDB stream large results with less memory
con.execute("INSTALL httpfs; LOAD httpfs;")

print("Querying and concatenating OmicIDX BioSamples... this may take a moment.")
con.execute(f"CREATE TABLE mapped AS {query}")
n_total = con.sql("SELECT count(*) FROM mapped").fetchone()[0]
print(f"Retrieved {n_total} samples with attributes.")

# ---------------------------------------------------------------------------
# 3. Filtering (in DuckDB)
# ---------------------------------------------------------------------------
# Drop rows where ALL mapped metadata columns are empty
any_metadata = " OR ".join(f"{c} IS NOT NULL" for c in metadata_cols)
con.execute(f"CREATE TABLE with_metadata AS SELECT * FROM mapped WHERE {any_metadata}")
con.execute("DROP TABLE mapped")  # free the space as soon as it is no longer needed
n_with_metadata = con.sql("SELECT count(*) FROM with_metadata").fetchone()[0]
print(f"{n_with_metadata} samples have at least one mapped metadata field.")

# Drop all non-bacteria BioSamples.
# bacteria_ids.json is a flat list of taxids; load it as a set of ints and register it as a table to join on
with open("data/bacteria_ids.json") as f:
    bacteria_ids = set(json.load(f))
con.register("bacteria_df", pd.DataFrame({"taxid": sorted(bacteria_ids)}))

con.execute("""
    CREATE TABLE bacterial AS
    SELECT * FROM with_metadata
    WHERE sample_taxon_id IN (SELECT taxid FROM bacteria_df)
""")
con.execute("DROP TABLE with_metadata")
n_bacterial = con.sql("SELECT count(*) FROM bacterial").fetchone()[0]
print(f"Dropped {n_with_metadata - n_bacterial} non-bacterial samples; {n_bacterial} remain.")

# ---------------------------------------------------------------------------
# 4. Verify mapping coverage (in DuckDB), then load the final table into pandas
# ---------------------------------------------------------------------------
# How many raw keys from the mapping table actually appear in the final (bacterial) samples
retrieved_keys = set(
    con.sql("SELECT DISTINCT unnest(matched_keys) AS k FROM bacterial").df()["k"]
)
mapping_keys = set(key_map["key"])
print(f"\n{len(retrieved_keys & mapping_keys)} of {len(mapping_keys)} mapping-table fields were retrieved.")
missing_keys = sorted(mapping_keys - retrieved_keys)
if missing_keys:
    print(f"Mapping fields never found ({len(missing_keys)}): {missing_keys}")

# matched_keys was only needed for the check above, so it is never loaded into pandas
df_filtered = con.sql("SELECT * EXCLUDE (matched_keys) FROM bacterial").df()
con.close()
shutil.rmtree(tmp_dir, ignore_errors=True)

# Verify the bacterial filter worked: every remaining taxid must be in the bacteria list
non_bacterial = df_filtered.loc[~df_filtered["sample_taxon_id"].isin(bacteria_ids), "sample_taxon_id"]
assert non_bacterial.empty, f"{len(non_bacterial)} non-bacterial samples remain, e.g. {non_bacterial.unique()[:10]}"
assert len(df_filtered) > 0, "Bacterial filter removed every sample - check bacteria_ids.json / taxid types"
print("Verified: all remaining samples have bacterial taxids.")

# Final column names must be unique, and should be exactly the base columns + one per label
expected_cols = ["biosample_accession", "sample_taxon_id", "sample_taxonomy_name"] + metadata_cols
assert df_filtered.columns.is_unique, f"Duplicate columns: {df_filtered.columns[df_filtered.columns.duplicated()].tolist()}"
assert list(df_filtered.columns) == expected_cols, f"Unexpected columns: {df_filtered.columns.tolist()}"
print(f"Verified: {len(df_filtered.columns)} unique field names: {df_filtered.columns.tolist()}")

print(f"\nSuccessfully extracted {len(df_filtered)} samples with relevant concatenated metadata.")
print("\nSample Data:")
print(df_filtered.head(10).to_string())

# ---------------------------------------------------------------------------
# 5. Coverage percentages (share of final samples with a non-null value per label)
# ---------------------------------------------------------------------------
print()
for col in metadata_cols:
    percent = df_filtered[col].notna().mean()
    print(f"{col}_percent percentage is {round(percent * 100, 3)}%")

df_filtered.to_parquet(OUTPUT_PATH)
print(f"\nSaved to {OUTPUT_PATH}")
