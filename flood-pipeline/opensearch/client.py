"""
OpenSearch Spatial Ingest and Query Handler for Flood Pipeline (Phase 4).

Features:
- Initializes index schemas for farmer_parcels (geo_point) and flood_zones (geo_shape).
- Seeds cadastral farmer parcel location records.
- Ingests predicted inundation polygons/multipolygons into flood_zones.
- Executes high-speed geo_shape spatial intersect queries to identify threatened plots.
- Produces clean recipient dispatch lists for Cedar policy verification and alerting.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("OpenSearchSpatialClient")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMAS_DIR = os.path.join(CURRENT_DIR, "schemas")
SAMPLE_PARCELS_PATH = os.path.join(CURRENT_DIR, "sample_parcels.json")

FARMER_INDEX = "farmer_parcels"
FLOOD_INDEX = "flood_zones"


class OpenSearchSpatialClient:
    """Client handling spatial indexing and polygon-plot intersect queries via OpenSearch REST API."""

    def __init__(self, endpoint: str = "http://localhost:9200", timeout: int = 10):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def ping(self) -> bool:
        """Check cluster availability and health."""
        try:
            resp = self.session.get(f"{self.endpoint}/_cluster/health", timeout=self.timeout)
            if resp.status_code == 200:
                health = resp.json()
                logger.info("Connected to OpenSearch cluster '%s' (status: %s)",
                            health.get("cluster_name"), health.get("status"))
                return True
        except requests.RequestException as err:
            logger.warning("OpenSearch ping failed at %s: %s", self.endpoint, err)
        return False

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        resp = self.session.head(f"{self.endpoint}/{index_name}", timeout=self.timeout)
        return resp.status_code == 200

    def delete_index(self, index_name: str) -> bool:
        """Delete an index if it exists."""
        if self.index_exists(index_name):
            resp = self.session.delete(f"{self.endpoint}/{index_name}", timeout=self.timeout)
            if resp.status_code == 200:
                logger.info("Deleted index '%s'", index_name)
                return True
            else:
                logger.error("Failed to delete index '%s': %s", index_name, resp.text)
                return False
        return True

    def create_index_from_schema(self, index_name: str, schema_path: str, force: bool = False) -> bool:
        """Create an index using a local schema definition JSON file."""
        if self.index_exists(index_name):
            if not force:
                logger.info("Index '%s' already exists. Skipping creation.", index_name)
                return True
            logger.info("Recreating index '%s'...", index_name)
            self.delete_index(index_name)

        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found at: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_body = json.load(f)

        resp = self.session.put(f"{self.endpoint}/{index_name}", json=schema_body, timeout=self.timeout)
        if resp.status_code in (200, 201):
            logger.info("Successfully created index '%s' from %s", index_name, os.path.basename(schema_path))
            return True
        else:
            logger.error("Failed to create index '%s': %s", index_name, resp.text)
            resp.raise_for_status()
            return False

    def init_schemas(self, force: bool = False) -> None:
        """Initialize both farmer_parcels and flood_zones schemas."""
        farmer_schema = os.path.join(SCHEMAS_DIR, "farmer_parcels.json")
        flood_schema = os.path.join(SCHEMAS_DIR, "flood_zones.json")

        self.create_index_from_schema(FARMER_INDEX, farmer_schema, force=force)
        self.create_index_from_schema(FLOOD_INDEX, flood_schema, force=force)

    def bulk_index_farmer_parcels(self, parcels: List[Dict[str, Any]]) -> int:
        """Bulk index cadastral parcel records with geo_point locations."""
        if not parcels:
            return 0

        lines = []
        for parcel in parcels:
            doc_id = parcel.get("farmer_id")
            action = {"index": {"_index": FARMER_INDEX, "_id": doc_id}}
            lines.append(json.dumps(action))
            lines.append(json.dumps(parcel))

        payload = "\n".join(lines) + "\n"
        headers = {"Content-Type": "application/x-ndjson"}
        resp = self.session.post(
            f"{self.endpoint}/_bulk?refresh=true",
            data=payload,
            headers=headers,
            timeout=self.timeout
        )
        resp.raise_for_status()
        res_data = resp.json()

        if res_data.get("errors"):
            logger.warning("Some items had errors during bulk indexing: %s", res_data)
        indexed_count = len(res_data.get("items", []))
        logger.info("Indexed %d farmer parcels into '%s'", indexed_count, FARMER_INDEX)
        return indexed_count

    def seed_parcels(self, file_path: Optional[str] = None) -> int:
        """Seed cadastral farmer parcels from sample JSON file."""
        target_path = file_path or SAMPLE_PARCELS_PATH
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Parcels data file not found at: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            parcels = json.load(f)

        return self.bulk_index_farmer_parcels(parcels)

    def index_flood_zone(
        self,
        alert_id: str,
        inundation_geometry: Dict[str, Any],
        max_depth_cm: float,
        time_to_peak_hours: int,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store predicted flood polygon/multipolygon geometry into flood_zones index.
        inundation_geometry must adhere to the GeoJSON Geometry specification (Polygon or MultiPolygon).
        """
        doc = {
            "alert_id": alert_id,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "max_depth_cm": float(max_depth_cm),
            "time_to_peak_hours": int(time_to_peak_hours),
            "inundation_geometry": inundation_geometry
        }

        resp = self.session.put(
            f"{self.endpoint}/{FLOOD_INDEX}/_doc/{alert_id}?refresh=true",
            json=doc,
            timeout=self.timeout
        )
        resp.raise_for_status()
        logger.info("Successfully indexed flood zone alert '%s' into '%s'", alert_id, FLOOD_INDEX)
        return resp.json()

    def find_threatened_parcels(
        self,
        inundation_geometry: Dict[str, Any],
        relation: str = "intersects",
        size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Execute high-speed spatial query against farmer_parcels using geo_shape query on plot_location.
        Relations: 'intersects' or 'within'.
        Returns list of threatened farmer records.
        """
        query = {
            "size": size,
            "query": {
                "geo_shape": {
                  "plot_location": {
                    "shape": inundation_geometry,
                    "relation": relation
                  }
                }
            }
        }

        resp = self.session.post(
            f"{self.endpoint}/{FARMER_INDEX}/_search",
            json=query,
            timeout=self.timeout
        )
        resp.raise_for_status()
        search_results = resp.json()

        hits = search_results.get("hits", {}).get("hits", [])
        threatened = [hit["_source"] for hit in hits]
        logger.info("Spatial intersect matched %d threatened parcel(s)", len(threatened))
        return threatened

    def find_threatened_parcels_by_indexed_shape(
        self,
        alert_id: str,
        relation: str = "intersects",
        size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Execute spatial match using a previously indexed flood zone document in flood_zones.
        """
        query = {
            "size": size,
            "query": {
                "geo_shape": {
                    "plot_location": {
                        "indexed_shape": {
                            "index": FLOOD_INDEX,
                            "id": alert_id,
                            "path": "inundation_geometry"
                        },
                        "relation": relation
                    }
                }
            }
        }

        resp = self.session.post(
            f"{self.endpoint}/{FARMER_INDEX}/_search",
            json=query,
            timeout=self.timeout
        )
        resp.raise_for_status()
        search_results = resp.json()

        hits = search_results.get("hits", {}).get("hits", [])
        threatened = [hit["_source"] for hit in hits]
        logger.info("Indexed shape spatial match for alert '%s' found %d parcel(s)", alert_id, len(threatened))
        return threatened


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenSearch Spatial Matching CLI for Flood Pipeline")
    parser.add_argument("--url", default="http://localhost:9200", help="OpenSearch base URL")
    parser.add_argument("--init", action="store_true", help="Initialize index schemas")
    parser.add_argument("--force", action="store_true", help="Recreate indices if they exist")
    parser.add_argument("--seed", action="store_true", help="Seed sample farmer parcels")
    parser.add_argument("--test", action="store_true", help="Run end-to-end spatial matching test")

    args = parser.parse_args()
    client = OpenSearchSpatialClient(endpoint=args.url)

    if not client.ping():
        logger.error("Could not reach OpenSearch at %s. Please make sure the service is up.", args.url)
        exit(1)

    if args.init or args.test:
        client.init_schemas(force=args.force)

    if args.seed or args.test:
        client.seed_parcels()

    if args.test:
        logger.info("Running spatial matching verification test...")
        # Define a test GeoJSON polygon covering Lowland Basin area (lat ~26.18 to 26.20, lon ~91.74 to 91.76)
        test_flood_polygon = {
            "type": "Polygon",
            "coordinates": [
                [
                    [91.7400, 26.1800],
                    [91.7620, 26.1800],
                    [91.7620, 26.2000],
                    [91.7400, 26.2000],
                    [91.7400, 26.1800]
                ]
            ]
        }

        test_alert_id = "ALERT_TEST_001"
        client.index_flood_zone(
            alert_id=test_alert_id,
            inundation_geometry=test_flood_polygon,
            max_depth_cm=28.5,
            time_to_peak_hours=3
        )

        matched = client.find_threatened_parcels(test_flood_polygon)
        print("\n--- SPATIAL MATCH RESULTS (Direct Geometry) ---")
        for p in matched:
            print(f" - [{p['farmer_id']}] {p['farmer_name']} ({p['sub_district']}) - Phone: {p['phone_number']} Location: {p['plot_location']}")

        indexed_matched = client.find_threatened_parcels_by_indexed_shape(test_alert_id)
        print("\n--- SPATIAL MATCH RESULTS (Pre-indexed Shape) ---")
        for p in indexed_matched:
            print(f" - [{p['farmer_id']}] {p['farmer_name']} ({p['sub_district']})")

        print(f"\nVerification successful: matched {len(matched)} parcels in flood polygon boundary.")
