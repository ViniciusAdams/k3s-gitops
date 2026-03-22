from opensearchpy import OpenSearch
import os
import json


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
    )


def main():
    index_name = get_required_env("SOURCE_INDEX")
    client = get_client()

    print("Starting ML pipeline job...")
    print(f"OpenSearch host: {os.getenv('OPENSEARCH_HOST')}")
    print(f"Index: {index_name}")

    print("Testing connection...")
    print(json.dumps(client.info(), indent=2, default=str))

    print(f"Reading from index: {index_name}")
    response = client.search(
        index=index_name,
        body={
            "size": 5,
            "query": {
                "match_all": {}
            }
        }
    )

    hits = response.get("hits", {}).get("hits", [])
    print(f"Fetched {len(hits)} documents")

    for i, hit in enumerate(hits, start=1):
        print(f"\nDocument {i}:")
        print(json.dumps(hit.get("_source", {}), indent=2, default=str))


if __name__ == "__main__":
    main()