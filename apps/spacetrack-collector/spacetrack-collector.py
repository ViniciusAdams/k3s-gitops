import os
import sys
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

import requests
import pandas as pd
import numpy as np
from opensearchpy import OpenSearch

#function to read enviroment variables
def get_env(name: str, default: str = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value

# reads an enviroment variable and converts into true ot false 
#example if the enviroment variable is set to "true", "1", "yes" or "y" it will return true, otherwise it will return false
def get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}

#reads the enviroment variable and converts it to an interger
def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)

# this is the part that talk with space track,login,session cookies query url and repeated http request
class SpaceTrackClient:
    LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
    BASE_QUERY_URL = "https://www.space-track.org/basicspacedata/query"

    def __init__(self, identity: str, password: str, timeout_login: int = 60, timeout_query: int = 180):
        self.identity = identity
        self.password = password
        self.timeout_login = timeout_login
        self.timeout_query = timeout_query
        self.session = requests.Session()
#when creating the client, it will automatically login to space track and store the session cookies for future requests
    def login(self) -> None:
        print("Logging in to Space-Track...")
        response = self.session.post(
            self.LOGIN_URL,
            data={
                "identity": self.identity,
                "password": self.password,
            },
            timeout=self.timeout_login,
        )
        response.raise_for_status()
        print("Login successful")
#building the query url for fetching GP data, it will use the lookback_days parameter to filter the data by epoch date
#/class/gp = fetch General Pertubations orbital records
# only return objects without adecay state so still in orbit
#
    def build_gp_query_url(self, lookback_days: int) -> str:
        return (
            f"{self.BASE_QUERY_URL}"
            f"/class/gp"
            f"/decay_date/null-val"
            f"/epoch/%3Enow-{lookback_days}"
            f"/orderby/norad_cat_id asc"
            f"/format/json"
        )
#fetch the data from space track using the built query url, it will return a list of dictionaries containing the GP data
    def fetch_gp_data(self, lookback_days: int = 1) -> List[Dict[str, Any]]:
        query_url = self.build_gp_query_url(lookback_days)
        print(f"Fetching GP data for last {lookback_days} day(s)...")
        response = self.session.get(query_url, timeout=self.timeout_query)
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, list):
            raise RuntimeError("Unexpected Space-Track response format: expected a list")

        print(f"Fetched {len(data)} records from Space-Track")
        return data

#Standardize the raw datagrame from Space-track
#this is the main cleaning step before Opensearch Indexing
def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert know date fieldst to real UTC datetimes
    date_columns = ["EPOCH", "LAUNCH_DATE", "DECAY_DATE", "CREATION_DATE"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Convert numeric fields
    numeric_columns = [
        "NORAD_CAT_ID",
        "GP_ID",
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
        "BSTAR",
        "ELEMENT_SET_NO",
        "REV_AT_EPOCH",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill string/categorical fields
    categorical_columns = [
        "OBJECT_NAME",
        "OBJECT_ID",
        "OBJECT_TYPE",
        "COUNTRY_CODE",
        "RCS_SIZE",
        "ORIGINATOR",
        "CENTER_NAME",
        "REF_FRAME",
        "TIME_SYSTEM",
        "CLASSIFICATION_TYPE",
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    # Stable id per orbital state
    if "NORAD_CAT_ID" in df.columns and "EPOCH" in df.columns:
        df["doc_id"] = (
            df["NORAD_CAT_ID"].astype("Int64").astype(str)
            + "-"
            + df["EPOCH"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").fillna("unknown")
        )
    else:
        raise ValueError("Required columns NORAD_CAT_ID and/or EPOCH missing from dataset")

    # Logical key
    if all(col in df.columns for col in ["NORAD_CAT_ID", "EPOCH", "GP_ID"]):
        df["DOC_KEY"] = (
            df["NORAD_CAT_ID"].astype("Int64").astype(str)
            + "|"
            + df["EPOCH"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").fillna("unknown")
            + "|"
            + df["GP_ID"].astype("Int64").astype(str)
        )
    else:
        df["DOC_KEY"] = (
            df["NORAD_CAT_ID"].astype("Int64").astype(str)
            + "|"
            + df["EPOCH"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ").fillna("unknown")
        )

    # Metadata for indexing
    df["@timestamp"] = df["EPOCH"]
    df["pipeline_name"] = "spacetrack-unified-ingest"
    df["source_system"] = "space-track"
    df["ingested_at"] = pd.Timestamp.now(tz="UTC")

    return df


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    safe_df = df.copy()

    for col in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    safe_df = safe_df.replace({np.nan: None})
    return safe_df.to_dict(orient="records")


def get_opensearch_client() -> OpenSearch:
    host = get_env("OPENSEARCH_HOST", required=True)
    port = int(get_env("OPENSEARCH_PORT", "9200"))
    user = get_env("OPENSEARCH_USER", required=True)
    password = get_env("OPENSEARCH_PASSWORD", required=True)

    use_ssl = get_bool_env("OPENSEARCH_USE_SSL", True)
    verify_certs = get_bool_env("OPENSEARCH_VERIFY_CERTS", False)

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(user, password),
        use_ssl=use_ssl,
        verify_certs=verify_certs,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )
    return client


def build_index_name() -> str:
    index_prefix = get_env("OPENSEARCH_INDEX_PREFIX", "spacetrack-raw")
    date_suffix = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    return f"{index_prefix}-{date_suffix}"


def create_bulk_payload(records: List[Dict[str, Any]], index_name: str) -> str:
    lines = []

    for record in records:
        doc_id = record["doc_id"]
        lines.append(json.dumps({"index": {"_index": index_name, "_id": doc_id}}))
        lines.append(json.dumps(record, default=str))

    return "\n".join(lines) + "\n"


def bulk_index_records(client: OpenSearch, records: List[Dict[str, Any]], index_name: str, batch_size: int = 500) -> None:
    total = len(records)
    indexed = 0

    for start in range(0, total, batch_size):
        batch = records[start:start + batch_size]
        payload = create_bulk_payload(batch, index_name)

        response = client.bulk(body=payload, request_timeout=120)

        if response.get("errors"):
            failed_items = [item for item in response.get("items", []) if "index" in item and item["index"].get("error")]
            raise RuntimeError(f"Bulk indexing failed for {len(failed_items)} items")

        indexed += len(batch)
        print(f"Indexed {indexed}/{total} documents into {index_name}")


def test_opensearch_connection(client: OpenSearch) -> None:
    print("Testing OpenSearch connection...")
    info = client.info()
    print("Connected to OpenSearch")
    print(json.dumps(info, indent=2, default=str))


def main() -> None:
    try:
        # Required envs
        space_track_identity = get_env("SPACE_TRACK_IDENTITY", required=True)
        space_track_password = get_env("SPACE_TRACK_PASSWORD", required=True)

        # Optional envs
        lookback_days = get_int_env("LOOKBACK_DAYS", 1)
        bulk_batch_size = get_int_env("BULK_BATCH_SIZE", 500)

        print("Starting Space-Track unified ingestion job")
        print(f"LOOKBACK_DAYS={lookback_days}")
        print(f"BULK_BATCH_SIZE={bulk_batch_size}")

        # 1. Fetch from Space-Track
        spacetrack = SpaceTrackClient(
            identity=space_track_identity,
            password=space_track_password,
        )
        spacetrack.login()
        raw_records = spacetrack.fetch_gp_data(lookback_days=lookback_days)

        if not raw_records:
            print("No records returned from Space-Track")
            return

        # 2. Standardize with pandas
        raw_df = pd.DataFrame(raw_records)
        print(f"Raw DataFrame shape: {raw_df.shape}")

        clean_df = standardize_dataframe(raw_df)
        print(f"Standardized DataFrame shape: {clean_df.shape}")

        # 3. Convert to records for OpenSearch
        records = dataframe_to_records(clean_df)

        # 4. Send to OpenSearch
        client = get_opensearch_client()
        test_opensearch_connection(client)

        index_name = build_index_name()
        print(f"Target index: {index_name}")

        bulk_index_records(
            client=client,
            records=records,
            index_name=index_name,
            batch_size=bulk_batch_size,
        )

        print("Ingestion completed successfully")
        print(f"Total indexed documents: {len(records)}")

    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()