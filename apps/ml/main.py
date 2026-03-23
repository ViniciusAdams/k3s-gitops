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


def save_documents(output_dir: Path, docs: list):
    json_path = output_dir / "raw_docs.json"
    jsonl_path = output_dir / "raw_docs.jsonl"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, default=str)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, default=str) + "\n")

    return json_path, jsonl_path


def build_dataframe(docs: list):
    rows = []
    for doc in docs:
        row = {"_id": doc.get("_id"), "_index": doc.get("_index")}
        source = doc.get("_source", {})

        if isinstance(source, dict):
            row.update(source)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def save_dataframe(output_dir: Path, df: pd.DataFrame):
    csv_path = output_dir / "raw_docs.csv"
    parquet_path = output_dir / "raw_docs.parquet"

    df.to_csv(csv_path, index=False)

    try:
        df.to_parquet(parquet_path, index=False)
        parquet_saved = True
    except Exception as e:
        print(f"Could not save parquet file: {e}")
        parquet_saved = False

    return csv_path, parquet_path if parquet_saved else None


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

        json_path, jsonl_path = save_documents(output_dir, docs)
        print(f"Saved JSON file to: {json_path}")
        print(f"Saved JSONL file to: {jsonl_path}")

        df = build_dataframe(docs)
        print(f"DataFrame created with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        csv_path, parquet_path = save_dataframe(output_dir, df)
        print(f"Saved CSV file to: {csv_path}")
        if parquet_path:
            print(f"Saved Parquet file to: {parquet_path}")

        print(f"Job completed successfully. Total documents fetched: {total_fetched}")

    except Exception as e:
        print(f"Job failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()