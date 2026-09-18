"""
Inference and Vectorization Engine for Hydrological Surrogate (Phase 3).

Features:
- Executes ONNX Runtime forward pass on 3-channel input tensor (<50ms).
- Identifies critical flood pixels (h >= 0.15m).
- Converts raster flood mask into vector geometries via rasterio.features.shapes.
- Filters out small single-pixel artifacts (area < 900 sq meters) before executing
  shapely.ops.unary_union, preventing CPU bottlenecks on fragmented water boundaries.
- Produces valid EPSG:4326 GeoJSON geometries for OpenSearch spatial indexing.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import onnxruntime as ort
from rasterio.features import shapes
from shapely.geometry import shape, mapping, MultiPolygon, Polygon
from shapely.ops import unary_union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MLInferenceEngine")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(CURRENT_DIR, "models", "flood_unet.onnx")

CRITICAL_DEPTH_THRESHOLD_M = 0.15  # 15 cm crop/property statutory threshold
MIN_FEATURE_AREA_M2 = 900.0        # Single 30m x 30m pixel area cutoff in sq meters


class FloodInundationInference:
    """Runs fast neural surrogate inference and vector extraction."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            from .export_onnx import build_physics_seeded_onnx
            build_physics_seeded_onnx(self.model_path)

        # Set thread affinity and CPU provider
        session_opts = ort.SessionOptions()
        session_opts.intra_op_num_threads = 4
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info("ONNX Runtime session initialized for model: %s", os.path.basename(self.model_path))

    def predict_depth_raster(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Runs neural inference over (1, 3, H, W) tensor stack.
        Returns predicted depth matrix (H, W) in meters and execution latency in ms.
        """
        t0 = time.perf_counter()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        depth_matrix = outputs[0][0, 0]  # Extract (H, W)
        max_depth = float(depth_matrix.max())
        mean_depth = float(depth_matrix.mean())

        logger.info("ONNX Inference completed in %.2f ms (Max Depth: %.3fm, Mean: %.3fm)",
                    latency_ms, max_depth, mean_depth)
        return depth_matrix, latency_ms

    def vectorize_inundation(
        self,
        depth_matrix: np.ndarray,
        transform: Any,
        depth_threshold_m: float = CRITICAL_DEPTH_THRESHOLD_M,
        min_area_m2: float = MIN_FEATURE_AREA_M2,
        center_lat: float = 17.98
    ) -> Dict[str, Any]:
        """
        Thresholds raster depths >= depth_threshold_m and extracts vector polygons.
        Filters out small single-pixel artifacts (area < 900 sq meters) before executing
        shapely.ops.unary_union to prevent CPU bottlenecks.
        """
        t0 = time.perf_counter()

        # Binary flood mask (uint8)
        flood_mask = (depth_matrix >= depth_threshold_m).astype(np.uint8)
        flooded_pixels = int(flood_mask.sum())

        if flooded_pixels == 0:
            logger.warning("No pixels exceeded critical depth threshold (%.2fm)", depth_threshold_m)
            return {"type": "MultiPolygon", "coordinates": []}

        # Conversion factor from deg^2 to m^2 at catchment latitude
        m_per_deg_lat = 111132.92
        m_per_deg_lon = 111412.84 * np.cos(np.radians(center_lat))
        sq_meters_per_sq_deg = m_per_deg_lat * m_per_deg_lon

        # Extract polygon shapes using rasterio
        raw_shapes = shapes(flood_mask, mask=(flood_mask == 1), transform=transform)

        valid_polygons: List[Polygon] = []
        filtered_count = 0

        for geom, val in raw_shapes:
            if val == 1:
                poly = shape(geom)
                # Filter small single-pixel artifacts (< 900 sq meters)
                area_m2 = poly.area * sq_meters_per_sq_deg
                if area_m2 >= min_area_m2:
                    valid_polygons.append(poly)
                else:
                    filtered_count += 1

        logger.info("Extracted %d valid inundation polygons (filtered out %d small single-pixel artifacts < %.0fm²)",
                    len(valid_polygons), filtered_count, min_area_m2)

        if not valid_polygons:
            logger.info("All extracted features were single-pixel noise. Returning empty geometry.")
            return {"type": "MultiPolygon", "coordinates": []}

        # Merge adjacent contiguous polygons
        t_union_0 = time.perf_counter()
        merged_geom = unary_union(valid_polygons)
        union_duration_ms = (time.perf_counter() - t_union_0) * 1000

        # Ensure MultiPolygon format for OpenSearch geo_shape compatibility
        if merged_geom.geom_type == "Polygon":
            final_geom = MultiPolygon([merged_geom])
        elif merged_geom.geom_type == "MultiPolygon":
            final_geom = merged_geom
        else:
            final_geom = MultiPolygon([p for p in merged_geom.geoms if p.geom_type == "Polygon"])

        geojson_geometry = mapping(final_geom)
        total_duration_ms = (time.perf_counter() - t0) * 1000

        logger.info("Vectorization complete in %.2f ms (unary_union: %.2f ms). Geometry type: %s with %d polygon part(s)",
                    total_duration_ms, union_duration_ms, geojson_geometry["type"], len(geojson_geometry["coordinates"]))

        return geojson_geometry


if __name__ == "__main__":
    from .terrain_stream import TerrainStreamEngine

    streamer = TerrainStreamEngine()
    tensor, tf = streamer.prepare_input_tensor()

    infer = FloodInundationInference()
    depth, lat_ms = infer.predict_depth_raster(tensor)
    geojson = infer.vectorize_inundation(depth, tf)

    print("Inference Latency: %.2f ms" % lat_ms)
    print("Max Water Depth:   %.2f cm" % (depth.max() * 100.0))
    print("GeoJSON Geometry Parts:", len(geojson["coordinates"]))
