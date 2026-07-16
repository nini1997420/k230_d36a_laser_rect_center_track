from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "digit_templates"
WIDTH = 48
HEIGHT = 72

FONTS = {
    "arial": Path(r"C:\Windows\Fonts\arial.ttf"),
    "arial_bold": Path(r"C:\Windows\Fonts\arialbd.ttf"),
    "calibri": Path(r"C:\Windows\Fonts\calibri.ttf"),
    "calibri_bold": Path(r"C:\Windows\Fonts\calibrib.ttf"),
    "segoe": Path(r"C:\Windows\Fonts\segoeui.ttf"),
    "segoe_bold": Path(r"C:\Windows\Fonts\segoeuib.ttf"),
}


def render_mask(digit: int, font_path: Path) -> Image.Image:
    canvas = Image.new("L", (240, 320), 0)
    font = ImageFont.truetype(str(font_path), 260)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), str(digit), font=font)
    x = (canvas.width - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (canvas.height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), str(digit), fill=255, font=font)

    content_box = canvas.getbbox()
    if content_box is None:
        raise RuntimeError(f"failed to render digit {digit}")
    glyph = canvas.crop(content_box)

    # OCR 检测框会紧贴字符；模板也采用紧轮廓，仅保留两像素安全边距。
    inner = glyph.resize((WIDTH - 4, HEIGHT - 4), Image.Resampling.LANCZOS)
    normalized = Image.new("L", (WIDTH, HEIGHT), 0)
    normalized.paste(inner, (2, 2))
    normalized = normalized.point(lambda value: 255 if value >= 112 else 0)
    return normalized


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old_file in OUTPUT.glob("digit_*.pgm"):
        old_file.unlink()

    preview = Image.new("RGB", (9 * 92, len(FONTS) * 100), "#20242b")
    preview_draw = ImageDraw.Draw(preview)

    for row, (font_name, font_path) in enumerate(FONTS.items()):
        if not font_path.exists():
            raise FileNotFoundError(font_path)
        preview_draw.text((4, row * 100 + 4), font_name, fill="white")
        for digit in range(1, 10):
            mask = render_mask(digit, font_path)
            mask.save(OUTPUT / f"digit_{digit}_{font_name}.pgm")

            shown = Image.new("RGB", mask.size, "white")
            black_digit = Image.eval(mask, lambda value: 255 - value)
            shown.paste(Image.merge("RGB", (black_digit, black_digit, black_digit)))
            shown = shown.resize((60, 90), Image.Resampling.NEAREST)
            preview.paste(shown, ((digit - 1) * 92 + 28, row * 100 + 8))

    # 另外输出九张便于人工查看的标准数字图片。
    primary_font = FONTS["segoe_bold"]
    for digit in range(1, 10):
        mask = render_mask(digit, primary_font)
        visible = Image.eval(mask, lambda value: 255 - value)
        visible.save(OUTPUT / f"digit_{digit}.png")

    preview.save(OUTPUT / "digit_templates_preview.png")
    print(f"generated {len(list(OUTPUT.glob('*.pgm')))} PGM templates in {OUTPUT}")


if __name__ == "__main__":
    main()
