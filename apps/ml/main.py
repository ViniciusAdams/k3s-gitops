from opensearchpy import OpenSearch
import os
import json
import sys


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
        timeout=30,          # socket timeout
        max_retries=2,
        retry_on_timeout=True,
    )


def main():
    try:
        index_name = get_required_env("SOURCE_INDEX")
        client = get_client()

        print("Starting ML pipeline job...")
        print(f"OpenSearch host: {os.getenv('OPENSEARCH_HOST')}")
        print(f"Index: {index_name}")

        print("Testing connection...")
        info = client.info(request_timeout=15)
        print(json.dumps(info, indent=2, default=str))

        batch_size = 100
        total_fetched = 0
        page = 0

        print(f"Reading from index: {index_name}")

        response = client.search(
            index=index_name,
            body={
                "size": batch_size,
                "sort": [{"_id": "asc"}],
                "query": {"match_all": {}}
            },
            request_timeout=30
        )

        hits = response.get("hits", {}).get("hits", [])

        while hits:
            page += 1
            print(f"Processing batch {page} with {len(hits)} documents")

            for hit in hits:
                total_fetched += 1
                source = hit.get("_source", {})
                doc_id = hit.get("_id", "unknown")
                print(f"Document {total_fetched} | ID: {doc_id}")
                print(json.dumps(source, indent=2, default=str))

            last_sort = hits[-1].get("sort")
            if not last_sort:
                break

            response = client.search(
                index=index_name,
                body={
                    "size": batch_size,
                    "sort": [{"_id": "asc"}],
                    "search_after": last_sort,
                    "query": {"match_all": {}}
                },
                request_timeout=30
            )
            hits = response.get("hits", {}).get("hits", [])

        print(f"Job completed successfully. Total documents fetched: {total_fetched}")

    except Exception as e:
        print(f"Job failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()