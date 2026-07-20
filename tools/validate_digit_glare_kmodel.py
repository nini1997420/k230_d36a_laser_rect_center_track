from __future__ import annotations

import argparse
from pathlib import Path

import nncase
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="digit_glare_int8.kmodel")
    args = parser.parse_args()

    images = np.load(args.model_dir / "validation_uint8.npy")
    labels = np.load(args.model_dir / "validation_labels.npy")
    simulator = nncase.Simulator()
    simulator.load_model((args.model_dir / args.model_name).read_bytes())

    matrix = np.zeros((10, 10), dtype=np.int32)
    correct = 0
    confidence_sum = 0.0
    threshold_stats = {
        (0.72, 0.18): [0, 0, 0],
        (0.80, 0.25): [0, 0, 0],
        (0.85, 0.30): [0, 0, 0],
    }
    for index, (image, expected) in enumerate(zip(images, labels)):
        simulator.set_input_tensor(
            0, nncase.RuntimeTensor.from_numpy(image[None].copy())
        )
        simulator.run()
        logits = simulator.get_output_tensor(0).to_numpy().reshape(-1)
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= np.sum(probabilities)
        predicted = int(np.argmax(probabilities))
        ordered = np.sort(probabilities)
        confidence = float(ordered[-1])
        gap = float(ordered[-1] - ordered[-2])
        matrix[int(expected), predicted] += 1
        correct += int(predicted == int(expected))
        confidence_sum += confidence
        for (min_confidence, min_gap), stats in threshold_stats.items():
            accepted = (predicted != 0 and confidence >= min_confidence
                        and gap >= min_gap)
            if accepted:
                stats[0] += 1
                stats[1] += int(predicted == int(expected))
                stats[2] += int(int(expected) == 0)
        if (index + 1) % 200 == 0:
            print("validated", index + 1)

    print("samples:", len(labels))
    print("kmodel accuracy:", correct / len(labels))
    print("mean top1 confidence:", confidence_sum / len(labels))
    print("threshold stats: conf/gap accepted correct false-positive-on-class0")
    for thresholds, stats in threshold_stats.items():
        print(thresholds, stats)
    print("confusion matrix rows=true cols=pred:")
    for row in matrix.tolist():
        print(row)


if __name__ == "__main__":
    main()
