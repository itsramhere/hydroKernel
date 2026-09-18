"""
Unit and Integration Test Suite for Phase 3 (Machine Learning Surrogate Inundation Engine).
"""

import os
import sys
import unittest
import numpy as np
from rasterio.transform import from_bounds

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ml_surrogate.inference import FloodInundationInference, MIN_FEATURE_AREA_M2
from ml_surrogate.export_onnx import build_physics_seeded_onnx


class TestPhase3MLEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        build_physics_seeded_onnx()
        cls.engine = FloodInundationInference()
        cls.transform = from_bounds(79.58, 17.96, 79.62, 18.00, 256, 256)

    def test_onnx_model_inference_latency(self):
        """Test neural inference execution time is strictly under 50 ms."""
        dummy_tensor = np.random.rand(1, 3, 256, 256).astype(np.float32)
        # Warmup
        self.engine.predict_depth_raster(dummy_tensor)

        depth, latency_ms = self.engine.predict_depth_raster(dummy_tensor)
        self.assertLess(latency_ms, 50.0, f"Inference took too long: {latency_ms:.2f} ms")
        self.assertEqual(depth.shape, (256, 256))

    def test_depression_accumulation_depths(self):
        """Test model predicts non-negative physically plausible water depths."""
        # Realistic valley tensor: low elevation in center, flat slope, high rain
        tensor = np.zeros((1, 3, 128, 128), dtype=np.float32)
        # Elevation: depression in center
        x = np.linspace(-1, 1, 128)
        y = np.linspace(-1, 1, 128)
        xx, yy = np.meshgrid(x, y)
        tensor[0, 0] = (xx**2 + yy**2).astype(np.float32)  # Low center (0.0), high edges (2.0)
        tensor[0, 1] = 0.05  # Flat slope in basin
        tensor[0, 2] = 85.0  # 85 mm rain storm

        depth, _ = self.engine.predict_depth_raster(tensor)
        self.assertGreaterEqual(float(depth.min()), 0.0, "Depth matrix must be non-negative")
        self.assertGreater(float(depth.max()), 0.15, "Depression must accumulate depth >= 15cm")

    def test_artifact_filtering_and_vectorization(self):
        """
        Verify that single-pixel artifacts (< 900 sq meters) are filtered out
        prior to unary_union to prevent CPU bottlenecks.
        """
        depth_matrix = np.zeros((256, 256), dtype=np.float32)

        # 1. Contiguous large flood basin (10x10 pixels = ~90,000 m^2)
        depth_matrix[100:110, 100:110] = 0.35

        # 2. Isolated single-pixel noise (< 900 m^2)
        depth_matrix[20, 20] = 0.50
        depth_matrix[45, 80] = 0.60
        depth_matrix[200, 50] = 0.40

        geojson = self.engine.vectorize_inundation(
            depth_matrix=depth_matrix,
            transform=self.transform,
            depth_threshold_m=0.15,
            min_area_m2=MIN_FEATURE_AREA_M2
        )

        self.assertIn(geojson["type"], ["Polygon", "MultiPolygon"])
        # The single-pixel artifacts should have been pruned, leaving only the main basin
        self.assertEqual(len(geojson["coordinates"]), 1)

    def test_geojson_validity(self):
        """Verify vector output geometry is valid GeoJSON MultiPolygon."""
        depth_matrix = np.full((128, 128), 0.25, dtype=np.float32)
        geojson = self.engine.vectorize_inundation(depth_matrix, self.transform)

        self.assertEqual(geojson["type"], "MultiPolygon")
        self.assertGreater(len(geojson["coordinates"]), 0)
        coords = geojson["coordinates"][0][0]
        # Check coordinates lie within Warangal bounds
        for pt in coords:
            self.assertGreaterEqual(pt[0], 79.57)
            self.assertLessEqual(pt[0], 79.63)
            self.assertGreaterEqual(pt[1], 17.95)
            self.assertLessEqual(pt[1], 18.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
