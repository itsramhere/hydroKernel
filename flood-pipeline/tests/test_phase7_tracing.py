"""
Unit and Integration Test Suite for Phase 7 (End-to-End Tracing & Observability with ADOT).
"""

import os
import sys
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from observability.tracer import setup_telemetry, trace_span
from main import run_pipeline
from dispatch.dispatcher import MockGatewayServer


class TestPhase7Observability(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tracer = setup_telemetry(prefer_grpc=True)
        cls.gateway = MockGatewayServer(port=8090)
        cls.gateway.start()

    @classmethod
    def tearDownClass(cls):
        cls.gateway.stop()

    def test_otel_span_instrumentation(self):
        """Test creating and finishing an individual OpenTelemetry span."""
        with trace_span("unit_test_trace_span", {"test.run": True}) as span:
            self.assertTrue(span.is_recording())

    def test_pipeline_all_spans_and_latency(self):
        """
        Verify end-to-end pipeline execution:
        - All 6 stages complete
        - Total runtime is well below 10 seconds threshold
        - OpenSearch matching, Cedar eval, and MicroVM dispatch succeed
        """
        result = run_pipeline(
            alert_id="ALERT_TEST_P7",
            system_mode="LIVE",
            gateway_url="http://127.0.0.1:8090",
            simulated_peak_hours=3
        )

        stages = result["stages"]
        required_stages = [
            "raster_ingest_window_ms",
            "ml_unet_inference_ms",
            "geojson_vectorize_ms",
            "opensearch_spatial_query_ms",
            "cedar_policy_evaluation_ms",
            "firecracker_microvm_dispatch_ms"
        ]

        for st in required_stages:
            self.assertIn(st, stages, f"Missing required stage: {st}")
            self.assertGreaterEqual(stages[st], 0.0)

        # Statutory SLA: Must complete in < 10,000 ms (10 seconds)
        self.assertLess(
            result["total_pipeline_ms"],
            10000.0,
            f"Pipeline exceeded latency limit: {result['total_pipeline_ms']}ms"
        )

        # Verification of state
        self.assertEqual(result["cedar_decision"], "ALLOW")
        self.assertGreater(result["threatened_parcels_count"], 0)
        self.assertGreater(result["dispatched_count"], 0)

    def test_dry_run_pipeline_tracing(self):
        """
        Test that in DRY_RUN mode:
        - Traces are still fully recorded
        - Cedar denies dispatch
        - No messages are dispatched
        """
        result = run_pipeline(
            alert_id="ALERT_TEST_DRYRUN",
            system_mode="DRY_RUN",
            gateway_url="http://127.0.0.1:8090",
            simulated_peak_hours=2
        )

        self.assertEqual(result["cedar_decision"], "DENY")
        self.assertEqual(result["dispatched_count"], 0)
        self.assertIn("cedar_policy_evaluation_ms", result["stages"])
        self.assertIn("firecracker_microvm_dispatch_ms", result["stages"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
