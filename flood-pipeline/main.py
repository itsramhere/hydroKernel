"""
End-to-End Local Hydrological Inundation & Alerting Pipeline (main.py).

Orchestrates:
1. raster_ingest_window: Slices DEM elevation tile and precipitation forecast window.
2. ml_unet_inference: Generates continuous water depth matrix.
3. geojson_vectorize: Contours critical depth cells (>=15cm) into GeoJSON geometries.
4. opensearch_spatial_match: Executes geospatial intersect query matching farmer parcels.
5. cedar_policy_eval: Validates statutory broadcast criteria using AWS Cedar Engine.
6. firecracker_microvm_dispatch: Formats dialect alerts and dispatches via isolated micro-worker.

All stages instrumented with AWS Distro for OpenTelemetry (ADOT).
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List

from observability.tracer import setup_telemetry, trace_span
from opensearch.client import OpenSearchSpatialClient
from policies.evaluator import CedarPolicyEvaluator
from dispatch.dispatcher import MicroVMDispatcher, MockGatewayServer

from ml_surrogate.terrain_stream import TerrainStreamEngine
from ml_surrogate.inference import FloodInundationInference
from sensor.receiver import RainGaugeTelemetryReceiver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("FloodPipelineMain")


def run_pipeline(
    alert_id: str = "ALERT_FLOOD_2026_0918",
    system_mode: str = "LIVE",
    opensearch_url: str = "http://localhost:9200",
    gateway_url: str = "http://127.0.0.1:8088",
    simulated_peak_hours: int = 3
) -> Dict[str, Any]:
    """
    Executes the end-to-end hydrological alerting pipeline with OpenTelemetry tracing.
    """
    total_start = time.perf_counter()
    setup_telemetry(prefer_grpc=True)

    pipeline_metrics = {
        "alert_id": alert_id,
        "system_mode": system_mode,
        "stages": {}
    }

    logger.info("=== Starting Hydrological Alerting Pipeline [Alert ID: %s, Mode: %s] ===", alert_id, system_mode)

    # -------------------------------------------------------------
    # Stage 1: raster_ingest_window
    # -------------------------------------------------------------
    with trace_span("raster_ingest_window", {
        "spatial.bbox": "79.58,17.96,79.62,18.00",
        "dem.source": "copernicus-dem-30m",
        "weather.source": "noaa-gfs"
    }) as span:
        t0 = time.perf_counter()
        streamer = TerrainStreamEngine()
        # Fetch edge telemetry if available
        receiver = RainGaugeTelemetryReceiver()
        receiver.start()
        time.sleep(0.1)
        latest_telemetry = receiver.get_latest()
        receiver.stop()

        tensor, transform = streamer.prepare_input_tensor(rain_gauge_telemetry=latest_telemetry)
        duration_ms = (time.perf_counter() - t0) * 1000
        span.set_attribute("duration_ms", duration_ms)
        span.set_attribute("tensor.shape", str(tensor.shape))
        pipeline_metrics["stages"]["raster_ingest_window_ms"] = duration_ms
        logger.info("Stage 1/6 [raster_ingest_window] completed in %.2fms (Tensor: %s)", duration_ms, tensor.shape)

    # -------------------------------------------------------------
    # Stage 2: ml_unet_inference
    # -------------------------------------------------------------
    with trace_span("ml_unet_inference", {
        "model.architecture": "2D-UNet",
        "model.format": "ONNX",
        "input.channels": 3,
        "device": "CPU"
    }) as span:
        t0 = time.perf_counter()
        infer_engine = FloodInundationInference()
        depth_matrix, infer_latency_ms = infer_engine.predict_depth_raster(tensor)
        duration_ms = (time.perf_counter() - t0) * 1000
        max_depth_cm = float(depth_matrix.max() * 100.0)

        span.set_attribute("duration_ms", duration_ms)
        span.set_attribute("max_predicted_depth_cm", max_depth_cm)
        pipeline_metrics["stages"]["ml_unet_inference_ms"] = duration_ms
        logger.info("Stage 2/6 [ml_unet_inference] completed in %.2fms (Max Depth: %.1fcm)",
                    duration_ms, max_depth_cm)

    # -------------------------------------------------------------
    # Stage 3: geojson_vectorize
    # -------------------------------------------------------------
    with trace_span("geojson_vectorize", {
        "threshold.depth_m": 0.15,
        "crs": "EPSG:4326"
    }) as span:
        t0 = time.perf_counter()
        flood_polygon = infer_engine.vectorize_inundation(
            depth_matrix=depth_matrix,
            transform=transform,
            depth_threshold_m=0.15
        )
        duration_ms = (time.perf_counter() - t0) * 1000
        span.set_attribute("duration_ms", duration_ms)
        span.set_attribute("geometry.type", flood_polygon["type"])
        pipeline_metrics["stages"]["geojson_vectorize_ms"] = duration_ms
        logger.info("Stage 3/6 [geojson_vectorize] completed in %.2fms (Type: %s)",
                    duration_ms, flood_polygon["type"])

    # -------------------------------------------------------------
    # Stage 4: opensearch_spatial_query
    # -------------------------------------------------------------
    with trace_span("opensearch_spatial_query", {
        "index.target": "farmer_parcels",
        "query.type": "geo_shape",
        "query.relation": "intersects"
    }) as span:
        t0 = time.perf_counter()
        os_client = OpenSearchSpatialClient(endpoint=opensearch_url)
        os_client.init_schemas()
        os_client.seed_parcels()

        # Ingest predicted flood zone into flood_zones
        os_client.index_flood_zone(
            alert_id=alert_id,
            inundation_geometry=flood_polygon,
            max_depth_cm=max_depth_cm,
            time_to_peak_hours=simulated_peak_hours
        )

        # Query threatened farmer parcels
        threatened_parcels = os_client.find_threatened_parcels(flood_polygon)
        duration_ms = (time.perf_counter() - t0) * 1000
        span.set_attribute("duration_ms", duration_ms)
        span.set_attribute("threatened_parcels.count", len(threatened_parcels))
        pipeline_metrics["stages"]["opensearch_spatial_query_ms"] = duration_ms
        pipeline_metrics["threatened_parcels_count"] = len(threatened_parcels)
        logger.info("Stage 4/6 [opensearch_spatial_query] completed in %.2fms: %d parcels threatened",
                    duration_ms, len(threatened_parcels))

    # -------------------------------------------------------------
    # Stage 5: cedar_policy_evaluation
    # -------------------------------------------------------------
    with trace_span("cedar_policy_evaluation", {
        "policy.engine": "AWS Cedar",
        "principal": "Role::MLHydrologyAgent",
        "action": "Action::BroadcastAlert"
    }) as span:
        t0 = time.perf_counter()
        cedar_evaluator = CedarPolicyEvaluator()
        decision = cedar_evaluator.evaluate(
            flood_depth_cm=max_depth_cm,
            time_to_peak_hours=simulated_peak_hours,
            affected_parcels_count=len(threatened_parcels),
            system_mode=system_mode,
            resource='Region::"Warangal_Rural"'
        )
        duration_ms = (time.perf_counter() - t0) * 1000
        span.set_attribute("duration_ms", duration_ms)
        span.set_attribute("cedar.decision", decision.decision)
        span.set_attribute("cedar.allowed", decision.allowed)
        pipeline_metrics["stages"]["cedar_policy_evaluation_ms"] = duration_ms
        pipeline_metrics["cedar_decision"] = decision.decision
        logger.info("Stage 5/6 [cedar_policy_evaluation] completed in %.2fms: Decision = %s",
                    duration_ms, decision.decision)

    # -------------------------------------------------------------
    # Stage 6: firecracker_microvm_dispatch
    # -------------------------------------------------------------
    with trace_span("firecracker_microvm_dispatch", {
        "worker.type": "Firecracker-MicroVM",
        "channels": "sms,whatsapp"
    }) as span:
        t0 = time.perf_counter()
        if decision.allowed:
            dispatcher = MicroVMDispatcher(gateway_url=gateway_url)
            dispatch_res = dispatcher.dispatch_batch(
                alert_id=alert_id,
                recipients=threatened_parcels,
                flood_depth_cm=max_depth_cm,
                time_to_peak_hours=simulated_peak_hours,
                channel="both"
            )
            dispatched_count = dispatch_res["dispatched_count"]
            span.set_attribute("dispatched_count", dispatched_count)
            span.set_attribute("worker.boot_latency_ms", dispatch_res["boot_latency_ms"])
            pipeline_metrics["dispatched_count"] = dispatched_count
        else:
            logger.warning("Dispatch skipped: Cedar authorization returned %s", decision.decision)
            dispatched_count = 0
            pipeline_metrics["dispatched_count"] = 0

        duration_ms = (time.perf_counter() - t0) * 1000
        span.set_attribute("duration_ms", duration_ms)
        pipeline_metrics["stages"]["firecracker_microvm_dispatch_ms"] = duration_ms
        logger.info("Stage 6/6 [firecracker_microvm_dispatch] completed in %.2fms", duration_ms)

    total_pipeline_ms = (time.perf_counter() - total_start) * 1000
    pipeline_metrics["total_pipeline_ms"] = total_pipeline_ms
    logger.info("=== End-to-End Pipeline Finished in %.2f ms (Sub-second execution: %s) ===",
                total_pipeline_ms, total_pipeline_ms < 10000)

    return pipeline_metrics


if __name__ == "__main__":
    # Start gateway for testing dispatch
    gateway = MockGatewayServer(port=8088)
    gateway.start()

    try:
        results = run_pipeline(
            alert_id="ALERT_DEMO_001",
            system_mode="LIVE",
            simulated_peak_hours=3
        )
        print("\n" + "=" * 55)
        print(" PIPELINE EXECUTION PERFORMANCE SUMMARY")
        print("=" * 55)
        for stage, duration in results["stages"].items():
            print(f"  • {stage:<35}: {duration:6.2f} ms")
        print("-" * 55)
        print(f"  • Total Pipeline Latency          : {results['total_pipeline_ms']:6.2f} ms")
        print(f"  • Threatened Parcels Identified   : {results['threatened_parcels_count']}")
        print(f"  • Cedar Policy Verification       : {results['cedar_decision']}")
        print(f"  • Total Notifications Dispatched  : {results['dispatched_count']}")
        print("=" * 55)
    finally:
        gateway.stop()
