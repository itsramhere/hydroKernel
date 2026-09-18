"""
Unit and Integration Test Suite for Phase 2 (Edge Telemetry & Open Terrain Ingestion).
"""

import os
import sys
import time
import json
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ml_surrogate.terrain_stream import TerrainStreamEngine, WARANGAL_BOUNDS
from sensor.receiver import RainGaugeTelemetryReceiver
import paho.mqtt.client as mqtt


class TestPhase2TelemetryAndStreaming(unittest.TestCase):

    def setUp(self):
        self.streamer = TerrainStreamEngine()

    def test_gdal_environment_flags(self):
        """Test GDAL performance flags are set for zero-egress single-request fetching."""
        self.assertEqual(os.environ.get("AWS_NO_SIGN_REQUEST"), "YES")
        self.assertEqual(os.environ.get("GDAL_DISABLE_READDIR_ON_OPEN"), "EMPTY_DIR")

    def test_dem_window_and_slope(self):
        """Test DEM window fetching and slope gradient computation."""
        dem, transform = self.streamer.fetch_dem_window()
        self.assertIsNotNone(dem)
        self.assertGreater(dem.shape[0], 0)
        self.assertGreater(dem.shape[1], 0)

        slope = self.streamer.compute_slope_gradient(dem)
        self.assertEqual(slope.shape, dem.shape)
        self.assertGreaterEqual(float(slope.min()), 0.0)
        self.assertLessEqual(float(slope.max()), 1.0)

    def test_precipitation_bias_correction(self):
        """Test ground-truth telemetry bias correction."""
        telemetry = {
            "station_id": "TEST_STATION",
            "rate_mm_hr": 45.0,
            "accum_6hr_mm": 97.5
        }
        grid = self.streamer.generate_precipitation_grid(
            dem_shape=(64, 64),
            base_forecast_mm=65.0,
            rain_gauge_telemetry=telemetry
        )
        # Factor is 97.5 / 65.0 = 1.5 -> grid should equal 97.5
        self.assertAlmostEqual(float(grid[0, 0]), 97.5, delta=0.1)

    def test_prepared_tensor_structure(self):
        """Test the (1, 3, H, W) tensor dimensions and value domains."""
        tensor, tf = self.streamer.prepare_input_tensor()
        self.assertEqual(len(tensor.shape), 4)
        self.assertEqual(tensor.shape[0], 1)
        self.assertEqual(tensor.shape[1], 3)
        self.assertGreaterEqual(float(tensor[0, 0].min()), 0.0)
        self.assertLessEqual(float(tensor[0, 0].max()), 1.0)

    def test_mqtt_telemetry_receipt(self):
        """Test publishing and receiving telemetry over local Mosquitto broker."""
        receiver = RainGaugeTelemetryReceiver()
        receiver.start()
        time.sleep(0.5)

        test_payload = {
            "station_id": "TEST_STATION_001",
            "rate_mm_hr": 42.0,
            "accum_6hr_mm": 84.0,
            "timestamp": int(time.time())
        }

        # Publish test message
        pub_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        try:
            pub_client.connect("localhost", 1883, keepalive=10)
            pub_client.publish("sensors/rainfall/TEST_STATION_001", json.dumps(test_payload))
            time.sleep(0.5)

            latest = receiver.get_latest()
            self.assertIsNotNone(latest)
            self.assertEqual(latest.get("station_id"), "TEST_STATION_001")
            self.assertEqual(latest.get("rate_mm_hr"), 42.0)
        finally:
            pub_client.disconnect()
            receiver.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
