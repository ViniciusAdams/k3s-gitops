from opensearchpy import OpenSearch
import os
import json
import sys
from pathlib import Path
import pandas as pd


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_client():
    host = get_required_env("OPENSEARCH_HOST")
    port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    user = get_required_env("OPENSEARCH_USER")
    password = get_required_env("OPENSEARCH_PASSWORD")

    use_ssl = os.getenv("USE_SSL", "false").lower() == "true"
    verify_certs = os.getenv("VERIFY_CERTS", "false").lower() == "true"

    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(user, password),
        use_ssl=use_ssl,
        verify_certs=verify_certs,
        timeout=30,
        max_retries=2,
        retry_on_timeout=True,
    )


def ensure_output_dir() -> Path:
    output_dir = Path(os.getenv("OUTPUT_DIR", "/data"))
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def fetch_all_documents(client, index_name: str, batch_size: int = 100):
    all_docs = []
    total_fetched = 0
    page = 0
    search_after = None

    print(f"Reading from index: {index_name}")

    while True:
        body = {
            "size": batch_size,
            "sort": [{"_id": "asc"}],
            "query": {"match_all": {}}
        }

        if search_after is not None:
            body["search_after"] = search_after

        response = client.search(
            index=index_name,
            body=body,
            request_timeout=30
        )

        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            break

        page += 1
        print(f"Processing batch {page} with {len(hits)} documents")

        for hit in hits:
            total_fetched += 1
            doc = {
                "_id": hit.get("_id", "unknown"),
                "_index": hit.get("_index", index_name),
                "_source": hit.get("_source", {})
            }
            all_docs.append(doc)

            if total_fetched % 25 == 0 or total_fetched == 1:
                print(f"Fetched {total_fetched} documents so far")

        search_after = hits[-1].get("sort")
        if not search_after:
            break

    return all_docs, total_fetched


def save_raw_documents(output_dir: Path, docs: list):
    raw_dir = output_dir / "raw"

    json_path = raw_dir / "raw_docs.json"
    jsonl_path = raw_dir / "raw_docs.jsonl"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, default=str)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, default=str) + "\n")

    return json_path, jsonl_path


def build_dataframe(docs: list) -> pd.DataFrame:
    rows = []

    for doc in docs:
        row = {
            "_id": doc.get("_id"),
            "_index": doc.get("_index")
        }

        source = doc.get("_source", {})
        if isinstance(source, dict):
            row.update(source)

        rows.append(row)

    return pd.DataFrame(rows)


def prepare_ml_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [
        "_id",
        "_index",
        "NORAD_CAT_ID",
        "OBJECT_NAME",
        "OBJECT_TYPE",
        "RCS_SIZE",
        "COUNTRY_CODE",
        "MEAN_MOTION",
        "ECCENTRICITY",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
        "SEMIMAJOR_AXIS",
        "APOAPSIS",
        "PERIAPSIS",
        "PERIOD",
        "EPOCH",
        "LAUNCH_DATE",
        "DECAY_DATE"
    ]

    existing_columns = [col for col in selected_columns if col in df.columns]
    ml_df = df[existing_columns].copy()

    numeric_columns = [
        "MEAN_MOTION",
        "ECCENTRICITY",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
        "SEMIMAJOR_AXIS",
        "APOAPSIS",
        "PERIAPSIS",
        "PERIOD",
        "NORAD_CAT_ID"
    ]

    for col in numeric_columns:
        if col in ml_df.columns:
            ml_df[col] = pd.to_numeric(ml_df[col], errors="coerce")

    date_columns = ["EPOCH", "LAUNCH_DATE", "DECAY_DATE"]
    for col in date_columns:
        if col in ml_df.columns:
            ml_df[col] = pd.to_datetime(ml_df[col], errors="coerce")

    required_numeric_columns = [
        "MEAN_MOTION",
        "ECCENTRICITY",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
        "SEMIMAJOR_AXIS",
        "APOAPSIS",
        "PERIAPSIS",
        "PERIOD"
    ]

    existing_required = [col for col in required_numeric_columns if col in ml_df.columns]
    ml_df = ml_df.dropna(subset=existing_required)

    categorical_columns = ["OBJECT_TYPE", "RCS_SIZE", "COUNTRY_CODE"]
    for col in categorical_columns:
        if col in ml_df.columns:
            ml_df[col] = ml_df[col].fillna("UNKNOWN")

    ml_df = ml_df.reset_index(drop=True)
    return ml_df


def save_dataframes(output_dir: Path, raw_df: pd.DataFrame, ml_df: pd.DataFrame):
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"

    raw_csv_path = raw_dir / "raw_docs.csv"
    raw_parquet_path = raw_dir / "raw_docs.parquet"
    ml_csv_path = processed_dir / "ml_ready_docs.csv"
    ml_parquet_path = processed_dir / "ml_ready_docs.parquet"

    raw_df.to_csv(raw_csv_path, index=False)
    ml_df.to_csv(ml_csv_path, index=False)

    raw_parquet_saved = False
    ml_parquet_saved = False

    try:
        raw_df.to_parquet(raw_parquet_path, index=False)
        raw_parquet_saved = True
    except Exception as e:
        print(f"Could not save raw parquet file: {e}")

    try:
        ml_df.to_parquet(ml_parquet_path, index=False)
        ml_parquet_saved = True
    except Exception as e:
        print(f"Could not save ML parquet file: {e}")

    return {
        "raw_csv": raw_csv_path,
        "raw_parquet": raw_parquet_path if raw_parquet_saved else None,
        "ml_csv": ml_csv_path,
        "ml_parquet": ml_parquet_path if ml_parquet_saved else None,
    }


def main():
    try:
        index_name = get_required_env("SOURCE_INDEX")
        client = get_client()
        output_dir = ensure_output_dir()
        batch_size = int(os.getenv("BATCH_SIZE", "100"))

        print("Starting ML pipeline job...")
        print(f"OpenSearch host: {os.getenv('OPENSEARCH_HOST')}")
        print(f"Index: {index_name}")
        print(f"Output directory: {output_dir}")

        print("Testing connection...")
        info = client.info(request_timeout=15)
        print(json.dumps(info, indent=2, default=str))

        docs, total_fetched = fetch_all_documents(client, index_name, batch_size=batch_size)

        if total_fetched == 0:
            print("No documents found.")
            return

        json_path, jsonl_path = save_raw_documents(output_dir, docs)
        print(f"Saved raw JSON file to: {json_path}")
        print(f"Saved raw JSONL file to: {jsonl_path}")

        raw_df = build_dataframe(docs)
        print(f"Raw DataFrame created with shape: {raw_df.shape}")

        ml_df = prepare_ml_dataframe(raw_df)
        print(f"ML-ready DataFrame created with shape: {ml_df.shape}")
        print(f"ML-ready columns: {list(ml_df.columns)}")

        saved_paths = save_dataframes(output_dir, raw_df, ml_df)
        print(f"Saved raw CSV file to: {saved_paths['raw_csv']}")
        if saved_paths["raw_parquet"]:
            print(f"Saved raw Parquet file to: {saved_paths['raw_parquet']}")

        print(f"Saved ML-ready CSV file to: {saved_paths['ml_csv']}")
        if saved_paths["ml_parquet"]:
            print(f"Saved ML-ready Parquet file to: {saved_paths['ml_parquet']}")

        print(f"Job completed successfully. Total documents fetched: {total_fetched}")
        print(f"Rows available for ML after cleaning: {len(ml_df)}")

    except Exception as e:
        print(f"Job failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()