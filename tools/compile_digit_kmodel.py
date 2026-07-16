from __future__ import annotations

import os
from pathlib import Path

import nncase
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "digit_cnn_model"
ONNX_PATH = MODEL_DIR / "digit_cnn.onnx"
KMODEL_PATH = MODEL_DIR / "digit_cnn_int8.kmodel"
DUMP_DIR = MODEL_DIR / "nncase_dump"


def main() -> None:
    if not ONNX_PATH.exists():
        raise FileNotFoundError(ONNX_PATH)
    calibration = np.load(MODEL_DIR / "calibration_uint8.npy")

    compile_options = nncase.CompileOptions()
    compile_options.target = "k230"
    compile_options.dump_ir = False
    compile_options.dump_asm = False
    compile_options.dump_dir = str(DUMP_DIR)

    import_options = nncase.ImportOptions()
    compiler = nncase.Compiler(compile_options)
    compiler.import_onnx(ONNX_PATH.read_bytes(), import_options)

    ptq_options = nncase.PTQTensorOptions()
    ptq_options.calibrate_method = "Kld"
    ptq_options.samples_count = len(calibration)
    calib_data = [[sample[None].copy() for sample in calibration]]
    ptq_options.set_tensor_data(calib_data)
    compiler.use_ptq(ptq_options)

    print("compiling for K230...")
    compiler.compile()
    KMODEL_PATH.write_bytes(compiler.gencode_tobytes())
    print("saved:", KMODEL_PATH)
    print("bytes:", KMODEL_PATH.stat().st_size)


if __name__ == "__main__":
    main()
