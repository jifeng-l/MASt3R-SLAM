# tensorrt_utils.py

import os
import torch
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
from typing import Tuple

class MASt3RTensorRTInference:
    """
    Simple wrapper to run inference with a serialized TensorRT engine.
    """
    def __init__(self, engine_path: str):
        import pycuda.driver as cuda, tensorrt as trt
        cuda.init()
        self.logger = TensorRTLogger()
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        import pycuda.driver as cuda
        # Allocate buffers
        output_shape = tuple(self.engine.get_binding_shape(1))
        d_input = cuda.mem_alloc(input_tensor.nbytes)
        d_output = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)
        # Copy input, execute, and copy back
        cuda.memcpy_htod(d_input, input_tensor)
        self.context.execute_v2(bindings=[int(d_input), int(d_output)])
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        return output

class TensorRTLogger(trt.ILogger):
    def __init__(self):
        super().__init__()

    def log(self, severity, msg):
        if severity == trt.Logger.ERROR:
            print(f"[ERROR] {msg}")
        elif severity == trt.Logger.WARNING:
            print(f"[WARNING] {msg}")
        elif severity == trt.Logger.INFO:
            print(f"[INFO] {msg}")
        elif severity == trt.Logger.VERBOSE:
            print(f"[VERBOSE] {msg}")

class MASt3RTensorRTConverter:
    def __init__(
        self,
        model_path: str,
        engine_path: str,
        precision: str = "fp16",
        max_workspace_size: int = 1 << 30,  # 1 GB
        max_batch_size: int = 1
    ):
        self.model_path = model_path
        self.engine_path = engine_path
        self.precision = precision
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size

        # --- TensorRT setup ---
        self.logger = TensorRTLogger()
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        self.config = self.builder.create_builder_config()

        # Precision flags
        if precision == "fp16" and self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8" and self.builder.platform_has_fast_int8:
            self.config.set_flag(trt.BuilderFlag.INT8)

        # Workspace size
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, self.max_workspace_size
        )

        # Set builder batch size (for old APIs; EXPLICIT_BATCH mode ignores it)
        self.builder.max_batch_size = self.max_batch_size

    def convert_mast3r_model(self, input_shape: Tuple[int,int,int,int]):
        # 1) Export to ONNX
        onnx_path = os.path.splitext(self.model_path)[0] + ".onnx"
        self._export_to_onnx(onnx_path, input_shape)

        # 2) Parse ONNX
        parser = trt.OnnxParser(self.network, self.logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parsing failed")

        # 3) Build TensorRT engine
        engine = self.builder.build_engine(self.network, self.config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # 4) Serialize engine to disk
        with open(self.engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"[INFO] TensorRT engine saved to {self.engine_path}")

    def _export_to_onnx(self, onnx_path: str, input_shape: Tuple[int,int,int,int]):
        import torch.nn as nn
        from mast3r_model import MASt3RModel

        # --- Load checkpoint ---
        ckpt = torch.load(self.model_path, map_location="cpu")
        print(f"[DEBUG] Loaded checkpoint of type {type(ckpt)}, keys = {list(ckpt.keys())}")

        # --- Figure out model vs. state_dict ---
        state_dict = None
        model: nn.Module = None

        if isinstance(ckpt, dict):
            # If someone saved the entire Module under "model"
            maybe_model = ckpt.get("model", None)
            if isinstance(maybe_model, nn.Module):
                model = maybe_model
                print("[DEBUG] Using ckpt['model'] as the module itself")
            # If "model" is actually a state-dict
            elif isinstance(maybe_model, dict):
                state_dict = maybe_model
                print("[DEBUG] Found state_dict under ckpt['model']")
            # If they used "state_dict" key
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state_dict = ckpt["state_dict"]
                print("[DEBUG] Found state_dict under ckpt['state_dict']")
            else:
                # fallback: assume entire ckpt is a state-dict
                state_dict = ckpt
                print("[DEBUG] Treating entire checkpoint as state_dict")
        else:
            raise RuntimeError(f"Unexpected checkpoint type: {type(ckpt)}")

        # --- Instantiate & load if needed ---
        if model is None:
            # build fresh model
            model = MASt3RModel(
                encoder_size="vit_large",
                decoder_size="base",
                # …any other args…
            )
            assert state_dict is not None, "No state_dict found to load!"
            model.load_state_dict(state_dict)
            print("[DEBUG] Loaded state_dict into fresh MASt3RModel()")

        # --- Now it's definitely a Module ---
        print(f"[DEBUG] Final model is {type(model)}, calling eval()")
        model.eval()

        # --- Dummy input for tracing ---
        dummy_input = torch.randn(input_shape)

        # --- Export to ONNX ---
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"},
                          "output": {0: "batch_size"}},
        )
        print(f"[INFO] ONNX model exported to {onnx_path}")

class MASt3RTensorRTInference:
    def __init__(self, engine_path: str):
        cuda.init()
        self.logger = TensorRTLogger()
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        # Allocate buffers
        input_size = input_tensor.nbytes
        output_shape = tuple(self.engine.get_binding_shape(1))
        output_size = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize

        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)

        # Copy to device
        cuda.memcpy_htod(d_input, input_tensor)

        # Execute
        self.context.execute_v2([int(d_input), int(d_output)])

        # Copy back
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        return output
