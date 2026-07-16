from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "digit_cnn_model"
IMAGE_SIZE = 32

FONT_NAMES = [
    "arial.ttf",
    "arialbd.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "segoeui.ttf",
    "segoeuib.ttf",
    "tahoma.ttf",
    "tahomabd.ttf",
    "verdana.ttf",
    "verdanab.ttf",
    "consola.ttf",
    "consolab.ttf",
    "bahnschrift.ttf",
    "times.ttf",
    "timesbd.ttf",
    "cambria.ttc",
    "cambriab.ttf",
    "georgia.ttf",
    "georgiab.ttf",
]
FONTS = [Path(r"C:\Windows\Fonts") / name for name in FONT_NAMES]
FONTS = [path for path in FONTS if path.exists()]


def normalize_glyph(mask: Image.Image, rng: random.Random) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    glyph = mask.crop(bbox)

    max_side = rng.randint(22, 27)
    scale = min(max_side / glyph.width, max_side / glyph.height)
    width = max(1, int(glyph.width * scale + 0.5))
    height = max(1, int(glyph.height * scale + 0.5))
    glyph = glyph.resize((width, height), Image.Resampling.BILINEAR)

    output = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    x = (IMAGE_SIZE - width) // 2 + rng.randint(-3, 3)
    y = (IMAGE_SIZE - height) // 2 + rng.randint(-3, 3)
    x = max(0, min(IMAGE_SIZE - width, x))
    y = max(0, min(IMAGE_SIZE - height, y))
    output.paste(glyph, (x, y))
    return output


def printed_digit(digit: int, rng: random.Random) -> np.ndarray:
    canvas = Image.new("L", (128, 128), 0)
    draw = ImageDraw.Draw(canvas)
    font_path = rng.choice(FONTS)
    font = ImageFont.truetype(str(font_path), rng.randint(76, 112))
    bbox = draw.textbbox((0, 0), str(digit), font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (128 - width) // 2 - bbox[0]
    y = (128 - height) // 2 - bbox[1]
    draw.text((x, y), str(digit), fill=rng.randint(205, 255), font=font)

    angle = rng.uniform(-16.0, 16.0)
    canvas = canvas.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    if rng.random() < 0.35:
        canvas = canvas.filter(
            ImageFilter.MaxFilter(3) if rng.random() < 0.65 else ImageFilter.MinFilter(3)
        )
    if rng.random() < 0.30:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.35, 1.0)))

    output = normalize_glyph(canvas, rng)
    array = np.asarray(output, dtype=np.float32)
    if rng.random() < 0.65:
        array += rng.normalvariate(0.0, 4.0) * np.random.randn(IMAGE_SIZE, IMAGE_SIZE)
    array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def negative_shape(rng: random.Random) -> np.ndarray:
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(canvas)
    kind = rng.randrange(7)
    color = rng.randint(180, 255)
    thickness = rng.randint(1, 4)

    if kind == 0:
        pass  # 完全空白的归一化输入
    elif kind == 1:
        # 纸张或手机边缘
        x = rng.choice([rng.randint(0, 5), rng.randint(26, 31)])
        draw.line((x, 0, x + rng.randint(-2, 2), 31), fill=color, width=thickness)
    elif kind == 2:
        y = rng.choice([rng.randint(0, 5), rng.randint(26, 31)])
        draw.line((0, y, 31, y + rng.randint(-2, 2)), fill=color, width=thickness)
    elif kind == 3:
        points = []
        for _ in range(rng.randint(2, 5)):
            points.append((rng.randint(0, 31), rng.randint(0, 31)))
        draw.line(points, fill=color, width=thickness)
    elif kind == 4:
        x1, y1 = rng.randint(0, 14), rng.randint(0, 14)
        x2, y2 = rng.randint(x1 + 5, 31), rng.randint(y1 + 5, 31)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)
    elif kind == 5:
        for _ in range(rng.randint(1, 5)):
            x = rng.randint(0, 28)
            y = rng.randint(0, 28)
            r = rng.randint(1, 4)
            draw.ellipse((x, y, x + r, y + r), fill=color)
    else:
        # 不完整的短竖线，与完整数字 1 区分
        x = rng.randint(6, 25)
        y1 = rng.randint(0, 14)
        y2 = min(31, y1 + rng.randint(5, 15))
        draw.line((x, y1, x + rng.randint(-2, 2), y2), fill=color, width=thickness)

    if rng.random() < 0.25:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    return np.asarray(canvas, dtype=np.uint8)


def build_dataset(samples_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    np.random.seed(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for _ in range(samples_per_class):
        images.append(negative_shape(rng))
        labels.append(0)
    for digit in range(1, 10):
        for _ in range(samples_per_class):
            images.append(printed_digit(digit, rng))
            labels.append(digit)

    order = np.random.permutation(len(images))
    x = np.stack(images, axis=0)[order]
    y = np.asarray(labels, dtype=np.int64)[order]
    return x[:, None, :, :], y


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 8 * 8, 96),
            nn.ReLU(inplace=False),
            nn.Dropout(0.15),
            nn.Linear(96, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32) / 255.0
        return self.classifier(self.features(x))


def accuracy(model: nn.Module, loader: DataLoader) -> tuple[float, list[list[int]]]:
    model.eval()
    correct = 0
    total = 0
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += labels.numel()
            for expected, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[expected][predicted] += 1
    return correct / max(1, total), matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--seed", type=int, default=230)
    args = parser.parse_args()

    if not FONTS:
        raise RuntimeError("no Windows fonts found")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("fonts:", len(FONTS))
    print("generating training data...")
    train_x, train_y = build_dataset(args.samples, args.seed)
    print("generating validation data...")
    val_x, val_y = build_dataset(max(200, args.samples // 5), args.seed + 1)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )

    torch.manual_seed(args.seed)
    model = DigitCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss) * labels.numel()
            seen += labels.numel()
        scheduler.step()
        val_accuracy, _ = accuracy(model, val_loader)
        print(
            "epoch %02d loss=%.5f val=%.4f"
            % (epoch, running_loss / seen, val_accuracy)
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no model")
    model.load_state_dict(best_state)
    val_accuracy, matrix = accuracy(model, val_loader)
    print("best validation accuracy:", val_accuracy)
    print("confusion matrix rows=true cols=pred:")
    for row in matrix:
        print(row)

    checkpoint = OUTPUT / "digit_cnn.pth"
    torch.save(model.state_dict(), checkpoint)

    onnx_path = OUTPUT / "digit_cnn.onnx"
    model.eval()
    dummy = torch.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=11,
        do_constant_folding=True,
        dynamo=False,
    )
    print("saved:", checkpoint)
    print("saved:", onnx_path)

    # nncase PTQ 量化使用已经归一化的真实合成样本。
    calibration = train_x[:100].copy()
    np.save(OUTPUT / "calibration_uint8.npy", calibration)
    np.save(OUTPUT / "validation_uint8.npy", val_x[:500])
    np.save(OUTPUT / "validation_labels.npy", val_y[:500])


if __name__ == "__main__":
    main()
