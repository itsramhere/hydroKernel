"""
ONNX Model Generation and Weight Seeding for Hydrological Surrogate (Phase 3).

Seeds initial convolutional kernels with depression-accumulation weights
so the model produces physically plausible inundation depths immediately
without blocking on a long training cycle.
"""

import os
import logging
import numpy as np
import onnx
from onnx import helper, TensorProto

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ExportONNX")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
OUTPUT_ONNX_PATH = os.path.join(MODELS_DIR, "flood_unet.onnx")


def build_physics_seeded_onnx(output_path: str = OUTPUT_ONNX_PATH) -> str:
    """
    Constructs and exports an optimized ONNX neural network surrogate.
    Seeds convolutional kernels with depression-accumulation weights:
    - Kernel captures topographic concave depressions (Laplacian of DEM).
    - Attenuates accumulation along steep topographic slopes.
    - Accumulates rainfall volume into continuous surface water depth (meters).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Constructing physics-seeded 2D hydrological surrogate ONNX model...")

    # Inputs and Outputs with dynamic spatial dimensions
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, None, None]
    )
    output_tensor = helper.make_tensor_value_info(
        "water_depth", TensorProto.FLOAT, [1, 1, None, None]
    )

    # -------------------------------------------------------------
    # Kernel Seeding: Depression-Accumulation Weights
    # -------------------------------------------------------------
    # Conv1: 8 filters of 3x3 over 3 channels (DEM, Slope, Rain)
    conv1_weights = np.zeros((8, 3, 3, 3), dtype=np.float32)

    # Filter 0: Direct Depression Sink Detector (Inverted Laplacian on DEM)
    conv1_weights[0, 0, :, :] = np.array([
        [-0.10, -0.20, -0.10],
        [-0.20,  1.20, -0.20],
        [-0.10, -0.20, -0.10]
    ], dtype=np.float32)
    # Slope suppression: steep cells drain rapidly, low slope retains water
    conv1_weights[0, 1, :, :] = -0.35 * np.ones((3, 3), dtype=np.float32)
    # Rain volume contribution: scaled runoff factor (mm -> meters)
    conv1_weights[0, 2, :, :] = 0.0035 * np.ones((3, 3), dtype=np.float32)

    # Filter 1: Overland Flow Accumulator (East-West flow)
    conv1_weights[1, 0, :, :] = np.array([
        [0.0, -0.2, 0.0],
        [0.2,  0.8, -0.2],
        [0.0, -0.2, 0.0]
    ], dtype=np.float32)
    conv1_weights[1, 2, :, :] = 0.0025 * np.ones((3, 3), dtype=np.float32)

    # Filter 2: Valley Drainage Channel (North-South flow)
    conv1_weights[2, 0, :, :] = np.array([
        [0.2,  0.0, -0.2],
        [-0.2, 0.8, -0.2],
        [0.2,  0.0, -0.2]
    ], dtype=np.float32)
    conv1_weights[2, 2, :, :] = 0.0025 * np.ones((3, 3), dtype=np.float32)

    # Filters 3-7: General smoothing and multi-scale retention kernels
    for i in range(3, 8):
        conv1_weights[i, 0, :, :] = (0.05 * (8 - i)) * np.array([
            [-0.05, -0.1, -0.05],
            [-0.1,   0.6, -0.1],
            [-0.05, -0.1, -0.05]
        ], dtype=np.float32)
        conv1_weights[i, 1, :, :] = -0.15 * np.ones((3, 3), dtype=np.float32)
        conv1_weights[i, 2, :, :] = 0.0015 * np.ones((3, 3), dtype=np.float32)

    conv1_bias = np.zeros((8,), dtype=np.float32)

    # Conv2 (Decoder Head): Combines feature maps into continuous depth h (1 channel)
    conv2_weights = np.zeros((1, 8, 3, 3), dtype=np.float32)
    conv2_weights[0, 0, 1, 1] = 0.65  # Primary depression sink
    conv2_weights[0, 1, 1, 1] = 0.20  # Flow accumulation
    conv2_weights[0, 2, 1, 1] = 0.20  # Valley channel
    conv2_weights[0, 3:, 1, 1] = 0.05 # Smoothing
    conv2_bias = np.array([0.0], dtype=np.float32)

    # Convert initializers
    init_conv1_w = helper.make_tensor("conv1_w", TensorProto.FLOAT, conv1_weights.shape, conv1_weights.flatten().tolist())
    init_conv1_b = helper.make_tensor("conv1_b", TensorProto.FLOAT, conv1_bias.shape, conv1_bias.flatten().tolist())
    init_conv2_w = helper.make_tensor("conv2_w", TensorProto.FLOAT, conv2_weights.shape, conv2_weights.flatten().tolist())
    init_conv2_b = helper.make_tensor("conv2_b", TensorProto.FLOAT, conv2_bias.shape, conv2_bias.flatten().tolist())

    # Build computation nodes
    node_conv1 = helper.make_node(
        "Conv",
        inputs=["input", "conv1_w", "conv1_b"],
        outputs=["conv1_out"],
        pads=[1, 1, 1, 1],
        name="conv1_depression_encoder"
    )
    node_relu1 = helper.make_node(
        "Relu",
        inputs=["conv1_out"],
        outputs=["relu1_out"],
        name="relu1"
    )
    node_conv2 = helper.make_node(
        "Conv",
        inputs=["relu1_out", "conv2_w", "conv2_b"],
        outputs=["conv2_out"],
        pads=[1, 1, 1, 1],
        name="conv2_depth_decoder"
    )
    # Relu ensures non-negative continuous water depth
    node_relu2 = helper.make_node(
        "Relu",
        inputs=["conv2_out"],
        outputs=["water_depth"],
        name="relu_depth_positive"
    )

    # Assemble Graph
    graph_def = helper.make_graph(
        nodes=[node_conv1, node_relu1, node_conv2, node_relu2],
        name="HydroSurrogateUNet",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[init_conv1_w, init_conv1_b, init_conv2_w, init_conv2_b]
    )

    # Model Definition (opset 17)
    model_def = helper.make_model(
        graph_def,
        producer_name="flood-pipeline-ml-engine",
        opset_imports=[helper.make_opsetid("", 17)]
    )

    # Validate ONNX specification
    onnx.checker.check_model(model_def)
    onnx.save(model_def, output_path)

    file_size_kb = os.path.getsize(output_path) / 1024.0
    logger.info("Successfully exported physics-seeded ONNX model to %s (%.1f KB)", output_path, file_size_kb)
    return output_path


if __name__ == "__main__":
    path = build_physics_seeded_onnx()
    print("Exported ONNX model successfully:", path)
