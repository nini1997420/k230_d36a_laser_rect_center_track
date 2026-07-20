from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


IMAGE_SIZE = 32
CONTENT_SIZE = 26

FONT_NAMES = [
    "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf",
    "segoeui.ttf", "segoeuib.ttf", "tahoma.ttf", "tahomabd.ttf",
    "verdana.ttf", "verdanab.ttf", "consola.ttf", "consolab.ttf",
    "bahnschrift.ttf", "times.ttf", "timesbd.ttf", "cambria.ttc",
    "cambriab.ttf", "georgia.ttf", "georgiab.ttf",
]
FONTS = [Path(r"C:\Windows\Fonts") / name for name in FONT_NAMES]
FONTS = [path for path in FONTS if path.exists()]


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(inplace=False),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 8 * 8, 96), nn.ReLU(inplace=False),
            nn.Dropout(0.15), nn.Linear(96, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.to(torch.float32) / 255.0))


def normalize_mask(mask: Image.Image, rng: random.Random | None = None) -> Image.Image:
    mask = mask.convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    glyph = mask.crop(bbox)
    max_side = CONTENT_SIZE if rng is None else rng.randint(23, 27)
    scale = min(max_side / max(1, glyph.width), max_side / max(1, glyph.height))
    width = max(1, int(glyph.width * scale + 0.5))
    height = max(1, int(glyph.height * scale + 0.5))
    glyph = glyph.resize((width, height), Image.Resampling.BILINEAR)
    output = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    jitter_x = 0 if rng is None else rng.randint(-2, 2)
    jitter_y = 0 if rng is None else rng.randint(-2, 2)
    x = max(0, min(IMAGE_SIZE - width, (IMAGE_SIZE - width) // 2 + jitter_x))
    y = max(0, min(IMAGE_SIZE - height, (IMAGE_SIZE - height) // 2 + jitter_y))
    output.paste(glyph, (x, y))
    return output


def render_font_digit(digit: int, rng: random.Random) -> Image.Image:
    canvas = Image.new("L", (128, 128), 0)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(rng.choice(FONTS)), rng.randint(76, 116))
    bbox = draw.textbbox((0, 0), str(digit), font=font,
                         stroke_width=rng.randint(0, 2))
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (128 - width) // 2 - bbox[0]
    y = (128 - height) // 2 - bbox[1]
    draw.text((x, y), str(digit), fill=255, font=font,
              stroke_width=rng.randint(0, 2), stroke_fill=255)
    angle = rng.uniform(-13.0, 13.0)
    canvas = canvas.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    return normalize_mask(canvas, rng)


def glare_erase(image: Image.Image, rng: random.Random) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    if rng.random() < 0.72:
        # Diagonal plastic-film highlight: erase a narrow band through the glyph.
        width = rng.randint(2, 6)
        x = rng.randint(5, 27)
        slope = rng.choice([-1, 1]) * rng.randint(5, 16)
        polygon = [
            (x - width, 0), (x + width, 0),
            (x + slope + width, 31), (x + slope - width, 31),
        ]
        draw.polygon(polygon, fill=0)
    if rng.random() < 0.35:
        x1 = rng.randint(5, 22)
        y1 = rng.randint(5, 23)
        w = rng.randint(3, 9)
        h = rng.randint(2, 8)
        draw.ellipse((x1, y1, x1 + w, y1 + h), fill=0)
    return image


def augment_positive(base: Image.Image, rng: random.Random,
                     stressed: bool = True) -> np.ndarray:
    image = base.copy()
    if rng.random() < 0.35:
        image = image.filter(
            ImageFilter.MaxFilter(3) if rng.random() < 0.65
            else ImageFilter.MinFilter(3)
        )
    if stressed:
        image = glare_erase(image, rng)
    if rng.random() < 0.30:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.25, 0.8)))
    if rng.random() < 0.25:
        # A small neutral shadow/crack fragment; the class must remain the digit.
        draw = ImageDraw.Draw(image)
        x1, y1 = rng.randint(0, 28), rng.randint(0, 28)
        x2, y2 = x1 + rng.randint(-5, 6), y1 + rng.randint(3, 11)
        draw.line((x1, y1, x2, y2), fill=rng.randint(90, 180), width=1)
    array = np.asarray(image, dtype=np.float32)
    array += np.random.normal(0.0, rng.uniform(0.0, 6.0), array.shape)
    return np.clip(array, 0, 255).astype(np.uint8)


def negative_shape(rng: random.Random) -> np.ndarray:
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(image)
    kind = rng.randrange(10)
    color = rng.randint(170, 255)
    thickness = rng.randint(1, 4)
    if kind == 0:
        pass
    elif kind in (1, 2):
        # The dominant field failure: a card edge or floor seam mistaken for 1.
        x = rng.choice([rng.randint(1, 7), rng.randint(11, 20), rng.randint(24, 30)])
        draw.line((x, rng.randint(0, 8), x + rng.randint(-2, 2),
                   rng.randint(23, 31)), fill=color, width=thickness)
    elif kind == 3:
        y = rng.choice([rng.randint(1, 7), rng.randint(24, 30)])
        draw.line((0, y, 31, y + rng.randint(-3, 3)), fill=color, width=thickness)
    elif kind == 4:
        draw.rectangle((rng.randint(0, 5), rng.randint(0, 5),
                        rng.randint(25, 31), rng.randint(25, 31)),
                       outline=color, width=thickness)
    elif kind == 5:
        points = [(rng.randint(0, 31), rng.randint(0, 31))
                  for _ in range(rng.randint(2, 5))]
        draw.line(points, fill=color, width=thickness)
    elif kind == 6:
        for _ in range(rng.randint(1, 6)):
            x, y = rng.randint(0, 29), rng.randint(0, 29)
            draw.ellipse((x, y, x + rng.randint(1, 4), y + rng.randint(1, 4)),
                         fill=color)
    else:
        # An incomplete digit fragment must be rejected instead of becoming 1/5/6.
        digit = rng.randint(1, 9)
        base = render_font_digit(digit, rng)
        crop_h = rng.randint(7, 16)
        crop_y = rng.randint(0, IMAGE_SIZE - crop_h)
        fragment = base.crop((0, crop_y, IMAGE_SIZE, crop_y + crop_h))
        if rng.random() < 0.5:
            image.paste(fragment, (0, rng.randint(0, IMAGE_SIZE - crop_h)))
        else:
            crop_w = rng.randint(5, 13)
            crop_x = rng.randint(0, IMAGE_SIZE - crop_w)
            vertical = base.crop((crop_x, 0, crop_x + crop_w, IMAGE_SIZE))
            image.paste(vertical, (rng.randint(0, IMAGE_SIZE - crop_w), 0))
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7)))
    return np.asarray(image, dtype=np.uint8)


REAL_CROPS = {
    "intersection_3_4.png": [
        (3, (70, 90, 205, 280)),
        (4, (270, 125, 425, 315)),
    ],
    "intersection_7_5_6_8.png": [
        (7, (15, 70, 110, 275)),
        (5, (165, 75, 270, 275)),
        (6, (360, 85, 500, 285)),
        (8, (520, 85, 625, 295)),
    ],
}


def extract_real_masks(real_dir: Path) -> dict[int, list[Image.Image]]:
    result: dict[int, list[Image.Image]] = {digit: [] for digit in range(1, 10)}
    for name, entries in REAL_CROPS.items():
        path = real_dir / name
        if not path.exists():
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        for digit, (x1, y1, x2, y2) in entries:
            crop = rgb[y1:y2, x1:x2]
            gray = crop.mean(axis=2)
            spread = crop.max(axis=2) - crop.min(axis=2)
            threshold = np.clip(gray.mean() * 0.80, 55.0, 145.0)
            ink = (gray <= threshold) & (spread <= 65.0)
            # Remove long colored/black card-border rows and columns before bbox.
            row_keep = ink.sum(axis=1) <= ink.shape[1] * 0.82
            col_keep = ink.sum(axis=0) <= ink.shape[0] * 0.92
            ink &= row_keep[:, None]
            ink &= col_keep[None, :]
            mask = Image.fromarray((ink.astype(np.uint8) * 255), mode="L")
            result[digit].append(normalize_mask(mask))
    return result


def build_dataset(samples_per_class: int, seed: int, real_masks: dict[int, list[Image.Image]],
                  stressed: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    np.random.seed(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []
    for _ in range(samples_per_class * 2):
        images.append(negative_shape(rng))
        labels.append(0)
    for digit in range(1, 10):
        for _ in range(samples_per_class):
            if real_masks[digit] and rng.random() < 0.28:
                base = rng.choice(real_masks[digit])
                # Small rotation/renormalization prevents memorizing one screenshot crop.
                base = base.rotate(rng.uniform(-4.0, 4.0),
                                   resample=Image.Resampling.BILINEAR, fillcolor=0)
                base = normalize_mask(base, rng)
            else:
                base = render_font_digit(digit, rng)
            images.append(augment_positive(base, rng, stressed=stressed))
            labels.append(digit)
    order = np.random.permutation(len(images))
    x = np.stack(images, axis=0)[order]
    y = np.asarray(labels, dtype=np.int64)[order]
    return x[:, None, :, :], y


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, list[list[int]]]:
    model.eval()
    correct = total = 0
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images).argmax(dim=1)
            correct += int((predicted == labels).sum())
            total += labels.numel()
            for expected, actual in zip(labels.tolist(), predicted.tolist()):
                matrix[expected][actual] += 1
    return correct / max(1, total), matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1600)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=230718)
    args = parser.parse_args()
    if not FONTS:
        raise RuntimeError("no Windows fonts found")
    args.output.mkdir(parents=True, exist_ok=True)
    real_masks = extract_real_masks(args.real_dir)
    print("real masks:", {key: len(value) for key, value in real_masks.items()})
    train_x, train_y = build_dataset(args.samples, args.seed, real_masks, stressed=True)
    val_x, val_y = build_dataset(max(260, args.samples // 5), args.seed + 1,
                                 real_masks, stressed=True)
    clean_x, clean_y = build_dataset(180, args.seed + 2, real_masks, stressed=False)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=256, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=512, shuffle=False, num_workers=0,
    )
    clean_loader = DataLoader(
        TensorDataset(torch.from_numpy(clean_x), torch.from_numpy(clean_y)),
        batch_size=512, shuffle=False, num_workers=0,
    )
    torch.manual_seed(args.seed)
    model = DigitCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = count = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * labels.numel()
            count += labels.numel()
        scheduler.step()
        accuracy, _ = evaluate(model, val_loader)
        print("epoch %02d loss=%.5f stress_val=%.4f" %
              (epoch, loss_sum / count, accuracy), flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().clone()
                          for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no best state")
    model.load_state_dict(best_state)
    stress_accuracy, matrix = evaluate(model, val_loader)
    clean_accuracy, _ = evaluate(model, clean_loader)
    print("best stress accuracy:", stress_accuracy)
    print("clean accuracy:", clean_accuracy)
    print("stress confusion rows=true cols=pred:")
    for row in matrix:
        print(row)

    torch.save(model.state_dict(), args.output / "digit_glare_cnn.pth")
    model.eval()
    onnx_path = args.output / "digit_glare_cnn.onnx"
    torch.onnx.export(
        model, torch.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8),
        onnx_path, input_names=["input"], output_names=["logits"],
        opset_version=11, do_constant_folding=True, dynamo=False,
    )
    np.save(args.output / "calibration_uint8.npy", train_x[:160])
    np.save(args.output / "validation_uint8.npy", val_x[:1000])
    np.save(args.output / "validation_labels.npy", val_y[:1000])
    print("saved:", args.output)


if __name__ == "__main__":
    main()
