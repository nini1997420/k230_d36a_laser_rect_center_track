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
OUTPUT = ROOT / "paper_digit_model"
CANVAS = 160
IMAGE_SIZE = 48

FONT_NAMES = [
    "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf",
    "segoeui.ttf", "segoeuib.ttf", "tahoma.ttf", "tahomabd.ttf",
    "verdana.ttf", "verdanab.ttf", "consola.ttf", "consolab.ttf",
    "bahnschrift.ttf", "times.ttf", "timesbd.ttf", "cambria.ttc",
    "cambriab.ttf", "georgia.ttf", "georgiab.ttf",
]
FONTS = [Path(r"C:\Windows\Fonts") / name for name in FONT_NAMES]
FONTS = [path for path in FONTS if path.exists()]


def camera_background(rng: random.Random) -> Image.Image:
    base = rng.randint(45, 190)
    gx = rng.uniform(-45.0, 45.0)
    gy = rng.uniform(-45.0, 45.0)
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS]
    array = base + gx * (xx / (CANVAS - 1) - 0.5) + gy * (
        yy / (CANVAS - 1) - 0.5
    )
    array += np.random.normal(0.0, rng.uniform(1.0, 7.0), array.shape)
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(0, 5)):
        shade = rng.randint(20, 210)
        if rng.random() < 0.5:
            y = rng.randint(0, CANVAS - 1)
            draw.line((0, y, CANVAS, y + rng.randint(-15, 15)), fill=shade,
                      width=rng.randint(1, 8))
        else:
            x1, y1 = rng.randint(-30, 130), rng.randint(-30, 130)
            x2, y2 = x1 + rng.randint(15, 90), y1 + rng.randint(15, 90)
            draw.rectangle((x1, y1, x2, y2), outline=shade,
                           width=rng.randint(1, 5))
    return image


def make_card(digit: int | None, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    width = rng.randint(82, 132)
    height = rng.randint(100, 150)
    paper = rng.randint(205, 255)
    card = Image.new("L", (width, height), paper)
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(card)

    # Slight paper illumination gradient and border are common in camera frames.
    if rng.random() < 0.65:
        edge = rng.randint(150, 225)
        draw.rectangle((0, 0, width - 1, height - 1), outline=edge,
                       width=rng.randint(1, 3))

    if digit is not None:
        font_path = rng.choice(FONTS)
        font = ImageFont.truetype(str(font_path), rng.randint(78, 124))
        text = str(digit)
        bbox = draw.textbbox((0, 0), text, font=font,
                             stroke_width=rng.randint(0, 2))
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        max_w = int(width * rng.uniform(0.48, 0.78))
        max_h = int(height * rng.uniform(0.58, 0.86))
        scale = min(1.0, max_w / max(1, tw), max_h / max(1, th))
        if scale < 0.999:
            font = ImageFont.truetype(str(font_path),
                                      max(32, int(font.size * scale)))
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (width - tw) // 2 - bbox[0] + rng.randint(-8, 8)
        y = (height - th) // 2 - bbox[1] + rng.randint(-8, 8)
        draw.text((x, y), text, fill=rng.randint(0, 55), font=font,
                  stroke_width=rng.randint(0, 2), stroke_fill=rng.randint(0, 45))
    return card, mask


def paste_card(scene: Image.Image, card: Image.Image, mask: Image.Image,
               rng: random.Random) -> None:
    angle = rng.uniform(-12.0, 12.0)
    card = card.rotate(angle, resample=Image.Resampling.BICUBIC,
                       expand=True, fillcolor=0)
    mask = mask.rotate(angle, resample=Image.Resampling.BILINEAR,
                       expand=True, fillcolor=0)
    x = (CANVAS - card.width) // 2 + rng.randint(-12, 12)
    y = (CANVAS - card.height) // 2 + rng.randint(-10, 10)
    # A broad dark offset approximates the card/hand shadow.
    if rng.random() < 0.8:
        shadow = Image.new("L", card.size, rng.randint(20, 110))
        scene.paste(shadow, (x + rng.randint(3, 10), y + rng.randint(3, 10)), mask)
    scene.paste(card, (x, y), mask)


def finish_scene(scene: Image.Image, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.70:
        scene = scene.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.1)))
    scene = scene.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(scene, dtype=np.float32)
    gain = rng.uniform(0.80, 1.18)
    offset = rng.uniform(-14.0, 14.0)
    array = array * gain + offset
    array += np.random.normal(0.0, rng.uniform(0.0, 5.0), array.shape)
    return np.clip(array, 0, 255).astype(np.uint8)


def positive_scene(digit: int, rng: random.Random) -> np.ndarray:
    scene = camera_background(rng)
    card, mask = make_card(digit, rng)
    paste_card(scene, card, mask, rng)
    return finish_scene(scene, rng)


def negative_scene(rng: random.Random) -> np.ndarray:
    scene = camera_background(rng)
    kind = rng.randrange(6)
    if kind in (1, 2, 3):
        card, mask = make_card(None, rng)
        if kind >= 2:
            draw = ImageDraw.Draw(card)
            color = rng.randint(0, 100)
            if kind == 2:
                # Lines and boxes, deliberately not shaped like a complete digit.
                for _ in range(rng.randint(1, 4)):
                    y = rng.randint(12, card.height - 12)
                    draw.line((8, y, card.width - 8, y + rng.randint(-5, 5)),
                              fill=color, width=rng.randint(1, 5))
            else:
                for _ in range(rng.randint(1, 5)):
                    x = rng.randint(5, card.width - 15)
                    y = rng.randint(5, card.height - 15)
                    r = rng.randint(3, 12)
                    draw.ellipse((x, y, x + r, y + r), fill=color)
        paste_card(scene, card, mask, rng)
    elif kind == 4:
        # Hand-like foreground blob without a card.
        draw = ImageDraw.Draw(scene)
        skin = rng.randint(65, 185)
        x, y = rng.randint(-25, 70), rng.randint(45, 125)
        draw.ellipse((x, y, x + rng.randint(65, 130), y + rng.randint(35, 85)),
                     fill=skin)
    elif kind == 5:
        # Partial paper crossing the border must stay negative.
        card, mask = make_card(None, rng)
        angle = rng.uniform(-20, 20)
        card = card.rotate(angle, expand=True, fillcolor=0)
        mask = mask.rotate(angle, expand=True, fillcolor=0)
        scene.paste(card, (rng.choice([-card.width // 2, 125]), rng.randint(-30, 80)), mask)
    return finish_scene(scene, rng)


def build_dataset(samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    np.random.seed(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []
    # More negative examples reduce false UART transmissions in an empty scene.
    for _ in range(samples * 2):
        images.append(negative_scene(rng))
        labels.append(0)
    for digit in range(1, 10):
        for _ in range(samples):
            images.append(positive_scene(digit, rng))
            labels.append(digit)
    order = np.random.permutation(len(images))
    x = np.stack(images)[order]
    y = np.asarray(labels, dtype=np.int64)[order]
    return x[:, None, :, :], y


class PaperDigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 6 * 6, 128), nn.ReLU(),
            nn.Dropout(0.12), nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x.to(torch.float32) / 255.0))


def score(model: nn.Module, loader: DataLoader) -> tuple[float, list[list[int]]]:
    model.eval()
    matrix = [[0] * 10 for _ in range(10)]
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images).argmax(1)
            correct += int((predicted == labels).sum())
            total += labels.numel()
            for expected, actual in zip(labels.tolist(), predicted.tolist()):
                matrix[expected][actual] += 1
    return correct / total, matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=700)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--seed", type=int, default=9230)
    args = parser.parse_args()
    if not FONTS:
        raise RuntimeError("no Windows fonts found")
    OUTPUT.mkdir(exist_ok=True)
    print("generating paper-scene training data")
    train_x, train_y = build_dataset(args.samples, args.seed)
    print("generating validation data")
    val_x, val_y = build_dataset(max(160, args.samples // 5), args.seed + 1)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x),
                              torch.from_numpy(train_y)), batch_size=192,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x),
                            torch.from_numpy(val_y)), batch_size=384,
                            num_workers=0)
    torch.manual_seed(args.seed)
    model = PaperDigitCNN()
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
        accuracy, _ = score(model, val_loader)
        print("epoch %02d loss=%.5f val=%.4f" %
              (epoch, loss_sum / count, accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    accuracy, matrix = score(model, val_loader)
    print("best validation accuracy", accuracy)
    for row in matrix:
        print(row)
    torch.save(model.state_dict(), OUTPUT / "paper_digit_cnn.pth")
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE,
                      dtype=torch.uint8), OUTPUT / "paper_digit_cnn.onnx",
                      input_names=["input"], output_names=["logits"],
                      opset_version=13, dynamo=False)
    np.save(OUTPUT / "calibration_uint8.npy", train_x[:120])
    np.save(OUTPUT / "validation_uint8.npy", val_x[:800])
    np.save(OUTPUT / "validation_labels.npy", val_y[:800])
    print("saved to", OUTPUT)


if __name__ == "__main__":
    main()
