import duckdb
import pandas as pd

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

print("Querying and concatenating OmicIDX BioSamples... this may take a moment.")

query = """
    SELECT 
        accession AS biosample_accession,
        
        -- THE SAMPLE ORGANISM (Standardized Top-Level Columns)
        taxon_id AS sample_taxon_id,
        taxonomy_name AS sample_taxonomy_name,
        
        -- Host Taxid (Custom User Metadata)
        NULLIF(array_to_string(list_transform(
            list_filter(attribute_recs, x -> lower(x.attribute_name) IN (
                'host taxid', 'host_taxid', 'host id', 'host taxonomy id', 'host taxon id', 'host_taxon_id'
            )), 
            y -> y.value
        ), '; '), '') AS host_taxid,
        
        -- Host Health
        NULLIF(array_to_string(list_transform(
            list_filter(attribute_recs, x -> lower(x.attribute_name) IN (
                'host health', 'host health disease status', 'host health state', 'health status of host', 'host_disease'
            )), 
            y -> y.value
        ), '; '), '') AS host_health,

        -- Host Name
        NULLIF(array_to_string(list_transform(
            list_filter(attribute_recs, x -> lower(x.attribute_name) IN (
                'hostindividual', 'host class', 'host organism', 'nat-host', 'host_species', 'host species', 'host', 'specific host', 'specific-host', 'host name'
            )), 
            y -> y.value
        ), '; '), '') AS host_name,

        -- Host Comments
        NULLIF(array_to_string(list_transform(
            list_filter(attribute_recs, x -> lower(x.attribute_name) IN (
                'host sex', 'host_age', 'host_sex'
            )), 
            y -> y.value
        ), '; '), '') AS host_comments,

        -- Isolation Comments
        NULLIF(array_to_string(list_transform(
            list_filter(attribute_recs, x -> lower(x.attribute_name) IN (
                'isolation site', 'collection point', 'isolation source', 'isolation and growth condition', 'subsrc_note', 'samp_collect_device', 'collection place', 'isolation-source', 'collected_by', 'collection location', 'collection_method', 'culture collection', 'isolation_source', 'sample collection device or method', 'culture-collection', 'isolation comments'
            )), 
            y -> y.value
        ), '; '), '') AS isolation_comments,

        -- Comments
        NULLIF(array_to_string(list_transform(
            list_filter(attribute_recs, x -> lower(x.attribute_name) IN (
                'field_notes', 'note', 'comment', 'isolation', 'sample comment', 'notes'
            )), 
            y -> y.value
        ), '; '), '') AS comments

    FROM read_parquet('https://data-omicidx.cancerdatasci.org/biosample/parquet/biosamples.parquet')
    
    WHERE attribute_recs IS NOT NULL
"""

df = con.sql(query).df()

# Target columns to check for all NaNs (excludes the accession and standardized taxon IDs)
metadata_cols = ['host_taxid', 'host_health', 'host_name', 'host_comments', 'isolation_comments', 'comments']

# Drop rows where ALL extracted custom metadata columns are NaN
df_filtered = df.dropna(subset=metadata_cols, how='all')

#Drop all non-bacteria BioSamples
bacteria_ids = pd.read_json(r"data/bacteria_ids.json")
bacteria_mask = df['sample_taxon_id'].isin(bacteria_ids)
df_filtered = df_filtered[bacteria_mask]

print(f"\nSuccessfully extracted {len(df_filtered)} samples with relevant concatenated metadata.")
print("\nSample Data:")
print(df_filtered.head(10).to_string())

host_percent = len(df['host_name'].dropna())/len(df)

host_health_percent = len(df['host_health'].dropna())/len(df)

host_taxid_percent = len(df['host_taxid'].dropna())/len(df)

host_comments_percent =len(df['host_comments'].dropna())/len(df)

isolation_comments_percent= len(df['isolation_comments'].dropna())/len(df)

comments_percent= len(df['comments'].dropna())/len(df)



print(f"host_percent percentage is {round(host_percent*100, 3)}%")

print(f"host_health_percent percentage is {round(host_health_percent*100, 3)}%")

print(f"host_taxid_percent percentage is {round(host_taxid_percent*100, 3)}%")

print(f"host_comments_percent percentage is {round(host_comments_percent*100, 3)}%")

print(f"isolation_comments_percent percentage is {round(isolation_comments_percent*100, 3)}%")

print(f"comments_percent percentage is {round(comments_percent*100, 3)}%")

df.to_parquet("data/Whole_BioSample.parquet")