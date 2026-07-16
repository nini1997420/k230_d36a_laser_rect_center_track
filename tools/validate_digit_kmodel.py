from __future__ import annotations

from pathlib import Path

import nncase
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "digit_cnn_model"


def main() -> None:
    images = np.load(MODEL_DIR / "validation_uint8.npy")
    labels = np.load(MODEL_DIR / "validation_labels.npy")
    simulator = nncase.Simulator()
    simulator.load_model((MODEL_DIR / "digit_cnn_int8.kmodel").read_bytes())

    matrix = [[0 for _ in range(10)] for _ in range(10)]
    correct = 0
    confidence_sum = 0.0
    for index, (image, expected) in enumerate(zip(images, labels)):
        input_data = image[None].copy()
        simulator.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(input_data))
        simulator.run()
        logits = simulator.get_output_tensor(0).to_numpy().reshape(-1)
        peak = float(np.max(logits))
        probabilities = np.exp(logits - peak)
        probabilities /= np.sum(probabilities)
        predicted = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted])
        matrix[int(expected)][predicted] += 1
        correct += int(predicted == int(expected))
        confidence_sum += confidence
        if (index + 1) % 100 == 0:
            print("validated", index + 1)

    print("kmodel accuracy:", correct / len(labels))
    print("mean top1 confidence:", confidence_sum / len(labels))
    print("confusion matrix rows=true cols=pred:")
    for row in matrix:
        print(row)


if __name__ == "__main__":
    main()
