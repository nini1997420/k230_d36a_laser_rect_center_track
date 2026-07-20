from __future__ import annotations

import argparse
from pathlib import Path

import nncase
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--target", default="k230")
    parser.add_argument("--output-name", default="digit_glare_int8.kmodel")
    args = parser.parse_args()
    onnx_path = args.model_dir / "digit_glare_cnn.onnx"
    output_path = args.model_dir / args.output_name
    calibration = np.load(args.model_dir / "calibration_uint8.npy")

    compile_options = nncase.CompileOptions()
    compile_options.target = args.target
    compile_options.dump_ir = False
    compile_options.dump_asm = False
    compile_options.dump_dir = str(args.model_dir / "nncase_dump")

    compiler = nncase.Compiler(compile_options)
    compiler.import_onnx(onnx_path.read_bytes(), nncase.ImportOptions())
    ptq_options = nncase.PTQTensorOptions()
    ptq_options.calibrate_method = "Kld"
    ptq_options.samples_count = len(calibration)
    ptq_options.set_tensor_data(
        [[sample[None].copy() for sample in calibration]]
    )
    compiler.use_ptq(ptq_options)
    print("compiling for", args.target)
    compiler.compile()
    output_path.write_bytes(compiler.gencode_tobytes())
    print("saved:", output_path)
    print("bytes:", output_path.stat().st_size)


if __name__ == "__main__":
    main()
