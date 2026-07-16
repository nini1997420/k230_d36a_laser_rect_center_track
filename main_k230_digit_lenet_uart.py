"""
K230 纸片单数字识别（专用 CNN KModel）

目标：浅色纸片上的深色单数字 1~9。

流程：
    5 帧学习空环境
      -> 固定区内提取深色数字轮廓
      -> 轮廓尺寸/位置/占比判断“数字确实存在”
      -> 保持比例缩放并居中到 32x32 图像
      -> 专用 0~9 数字 CNN（0 为无数字/干扰）
      -> 完整观察 4 帧，至少 3 帧一致
      -> UART3 握手

不使用 OCR，不使用模板文件。
"""

from libs.PipeLine import PipeLine
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import os
import time
import gc

from machine import FPIOA, UART


# ------------------------- 画面 -------------------------
AI_W = 640
AI_H = 480
DISPLAY_W = 640
DISPLAY_H = 480
DISPLAY_MODE = "lcd"

# 纸片数字应完整放在此区域。
ROI_W = 200
ROI_H = 200
ROI_X = (AI_W - ROI_W) // 2
ROI_Y = (AI_H - ROI_H) // 2


# ------------------------- 环境和轮廓存在判断 -------------------------
WARMUP_FRAMES = 5
SCENE_MEAN_DELTA_MIN = 5.0
MIN_CONTRAST = 45.0
INK_THRESHOLD_RATIO = 0.42

EDGE_IGNORE_X = 12
EDGE_IGNORE_Y = 10
MIN_DIGIT_W = 8
MIN_DIGIT_H = 55
MAX_DIGIT_W = 165
MAX_DIGIT_H = 185
MIN_DIGIT_ASPECT = 0.07
MAX_DIGIT_ASPECT = 1.20
MIN_INK_RATIO = 0.012
MAX_INK_RATIO = 0.38
MAX_CENTER_DX = 55
MAX_CENTER_DY = 55


# ------------------------- 专用 CNN 和 4 帧确认 -------------------------
DIGIT_KMODEL = "/sdcard/kmodel/digit_cnn_int8.kmodel"
MODEL_SIZE = 32
MODEL_CONTENT_SIZE = 26
MIN_CNN_CONFIDENCE = 0.68
MIN_CNN_GAP = 0.15
CONFIRM_FRAMES = 3
CONFIRM_VOTES = 2


# ------------------------- UART3 -------------------------
UART_TX_PIN = 32
UART_RX_PIN = 33
UART_ID = 3
UART_BAUD = 115200
HANDSHAKE_RETRY_MS = 1000
HANDSHAKE_POLL_MS = 20
HANDSHAKE_DONE_DISPLAY_MS = 800

DEBUG_INTERVAL = 5


class UARTLink:
    def __init__(self):
        self.uart = None
        self.buffer = ""
        try:
            self.fpioa = FPIOA()
            self.fpioa.set_function(UART_TX_PIN, FPIOA.UART3_TXD, ie=0, oe=1)
            self.fpioa.set_function(UART_RX_PIN, FPIOA.UART3_RXD, ie=1, oe=0)
            uart_id = getattr(UART, "UART3", UART_ID)
            self.uart = UART(uart_id, baudrate=int(UART_BAUD))
            print(
                "UART ready: GPIO%d(TX) GPIO%d(RX) UART%d @ %d"
                % (UART_TX_PIN, UART_RX_PIN, UART_ID, UART_BAUD)
            )
        except Exception as exc:
            print("UART init failed:", exc)
            self.uart = None

    def _send(self, packet):
        if self.uart is None:
            return False
        self.uart.write((packet + "\r\n").encode())
        print("TX:", packet)
        return True

    def send_digit(self, digit):
        return self._send("[num,%d]" % int(digit))

    def send_confirmed(self, digit):
        return self._send("[num,confirmed,%d]" % int(digit))

    def read_packets(self):
        packets = []
        if self.uart is None:
            return packets
        try:
            available = self.uart.any()
            if available:
                data = self.uart.read(available)
                if data:
                    self.buffer += data.decode().replace("\r", "").replace("\n", "")
        except Exception as exc:
            print("UART RX error:", exc)
            return packets

        while True:
            start = self.buffer.find("[")
            if start < 0:
                if len(self.buffer) > 128:
                    self.buffer = ""
                break
            end = self.buffer.find("]", start + 1)
            if end < 0:
                self.buffer = self.buffer[start:]
                break
            content = self.buffer[start + 1 : end]
            self.buffer = self.buffer[end + 1 :]
            parts = [part.strip() for part in content.split(",")]
            if parts:
                packets.append(parts)
        return packets

    def close(self):
        if self.uart is not None:
            try:
                self.uart.deinit()
            except Exception:
                pass
            self.uart = None


class EmptySceneGate:
    def __init__(self):
        self.count = 0
        self.baseline_mean = 0.0
        self.ready = False
        self.delta = 0.0

    def update(self, roi_gray):
        current = float(np.mean(roi_gray))
        if not self.ready:
            self.baseline_mean = (
                self.baseline_mean * self.count + current
            ) / (self.count + 1)
            self.count += 1
            if self.count >= WARMUP_FRAMES:
                self.ready = True
                print("EMPTY SCENE READY: 5 frames learned")
            return False
        self.delta = abs(current - self.baseline_mean)
        return self.delta >= SCENE_MEAN_DELTA_MIN

    def status(self):
        if not self.ready:
            return "LEARN EMPTY %d/%d" % (self.count, WARMUP_FRAMES)
        return "SCENE DELTA %.1f" % self.delta


def projection_bounds(counts, minimum):
    start = -1
    end = -1
    for index in range(len(counts)):
        if float(counts[index]) >= minimum:
            if start < 0:
                start = index
            end = index + 1
    return start, end


def extract_digit_mask(frame_chw):
    """提取纸片上的深色数字，返回二值 mask、外包围和诊断信息。"""
    roi = frame_chw[
        :,
        ROI_Y : ROI_Y + ROI_H,
        ROI_X : ROI_X + ROI_W,
    ]
    gray = roi[0] / 3.0 + roi[1] / 3.0 + roi[2] / 3.0
    darkest = float(np.min(gray))
    brightest = float(np.max(gray))
    contrast = brightest - darkest
    if contrast < MIN_CONTRAST:
        return gray, None, None, "LOW CONTRAST %.1f" % contrast

    # 对浅色纸张上的深色数字，只保留亮度较低的笔画。
    threshold = darkest + contrast * INK_THRESHOLD_RATIO
    ink = gray < threshold
    inner = ink[
        EDGE_IGNORE_Y : ROI_H - EDGE_IGNORE_Y,
        EDGE_IGNORE_X : ROI_W - EDGE_IGNORE_X,
    ]
    ink_ratio = float(np.sum(inner)) / (
        (ROI_H - EDGE_IGNORE_Y * 2) * (ROI_W - EDGE_IGNORE_X * 2)
    )
    if ink_ratio < MIN_INK_RATIO or ink_ratio > MAX_INK_RATIO:
        return gray, ink, None, "INK RATIO %.3f" % ink_ratio

    row_counts = [0.0] * ROI_H
    for y in range(EDGE_IGNORE_Y, ROI_H - EDGE_IGNORE_Y):
        row_counts[y] = float(
            np.sum(ink[y, EDGE_IGNORE_X : ROI_W - EDGE_IGNORE_X])
        )
    y1, y2 = projection_bounds(row_counts, 3.0)
    if y1 < 0:
        return gray, ink, None, "NO DIGIT ROWS"

    col_counts = [0.0] * ROI_W
    for x in range(EDGE_IGNORE_X, ROI_W - EDGE_IGNORE_X):
        col_counts[x] = float(np.sum(ink[y1:y2, x]))
    x1, x2 = projection_bounds(col_counts, 3.0)
    if x1 < 0:
        return gray, ink, None, "NO DIGIT COLS"

    x1 = max(EDGE_IGNORE_X, x1 - 2)
    y1 = max(EDGE_IGNORE_Y, y1 - 2)
    x2 = min(ROI_W - EDGE_IGNORE_X, x2 + 2)
    y2 = min(ROI_H - EDGE_IGNORE_Y, y2 + 2)
    width = x2 - x1
    height = y2 - y1
    aspect = width / height
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5

    if (
        width < MIN_DIGIT_W
        or height < MIN_DIGIT_H
        or width > MAX_DIGIT_W
        or height > MAX_DIGIT_H
        or aspect < MIN_DIGIT_ASPECT
        or aspect > MAX_DIGIT_ASPECT
    ):
        return gray, ink, None, "BAD SIZE %dx%d" % (width, height)
    if (
        abs(center_x - ROI_W * 0.5) > MAX_CENTER_DX
        or abs(center_y - ROI_H * 0.5) > MAX_CENTER_DY
    ):
        return gray, ink, None, "OFF CENTER %.0f,%.0f" % (center_x, center_y)

    return gray, ink, (x1, y1, x2, y2), "DIGIT PRESENT"


def make_model_image(ink, box):
    """将数字保持比例缩放到 26x26 内，居中放入 32x32。"""
    x1, y1, x2, y2 = box
    src_w = x2 - x1
    src_h = y2 - y1
    scale = min(MODEL_CONTENT_SIZE / src_w, MODEL_CONTENT_SIZE / src_h)
    dst_w = max(1, int(src_w * scale + 0.5))
    dst_h = max(1, int(src_h * scale + 0.5))
    offset_x = (MODEL_SIZE - dst_w) // 2
    offset_y = (MODEL_SIZE - dst_h) // 2

    normalized = np.zeros((1, 1, MODEL_SIZE, MODEL_SIZE), dtype=np.uint8)
    for dy in range(dst_h):
        sy = y1 + min(src_h - 1, int(dy * src_h / dst_h))
        for dx in range(dst_w):
            sx = x1 + min(src_w - 1, int(dx * src_w / dst_w))
            if ink[sy, sx]:
                normalized[0, 0, offset_y + dy, offset_x + dx] = 255
    return normalized


class DigitKModelRecognizer:
    def __init__(self):
        self.kpu = nn.kpu()
        self.kpu.load_kmodel(DIGIT_KMODEL)
        print("Dedicated digit CNN ready:", DIGIT_KMODEL)

    def run(self, normalized):
        input_tensor = nn.from_numpy(normalized)
        self.kpu.set_input_tensor(0, input_tensor)
        self.kpu.run()
        output_tensor = self.kpu.get_output_tensor(0)
        logits = output_tensor.to_numpy().reshape((-1))

        peak = float(np.max(logits))
        exp_values = np.exp(logits - peak)
        probabilities = exp_values / float(np.sum(exp_values))
        best_class = int(np.argmax(probabilities))
        best_probability = float(probabilities[best_class])
        second_probability = 0.0
        for class_index in range(10):
            if class_index == best_class:
                continue
            value = float(probabilities[class_index])
            if value > second_probability:
                second_probability = value
        gap = best_probability - second_probability

        del input_tensor
        del output_tensor
        del logits
        del exp_values
        del probabilities

        # 类别 0 专门表示空白、纸张边缘、线条和杂点干扰。
        if (
            best_class == 0
            or best_probability < MIN_CNN_CONFIDENCE
            or gap < MIN_CNN_GAP
        ):
            return None, best_probability, gap, best_class
        return best_class, best_probability, gap, best_class

    def deinit(self):
        del self.kpu
        nn.shrink_memory_pool()


class FourFrameConfirmer:
    def __init__(self):
        self.history = []

    def update(self, digit):
        # 无数字时不启动计时；首个有效结果开始 4 帧窗口。
        if digit is None and not self.history:
            return None
        self.history.append(-1 if digit is None else int(digit))
        if len(self.history) > CONFIRM_FRAMES:
            self.history.pop(0)
        if len(self.history) < CONFIRM_FRAMES:
            return None

        winner = None
        winner_votes = 0
        for value in range(1, 10):
            votes = 0
            for item in self.history:
                if item == value:
                    votes += 1
            if votes > winner_votes:
                winner_votes = votes
                winner = value
        return winner if winner_votes >= CONFIRM_VOTES else None

    def status(self):
        return "CONFIRM FRAME %d/%d" % (len(self.history), CONFIRM_FRAMES)


def display_x(pl, x):
    return int(x * pl.display_size[0] / AI_W)


def display_y(pl, y):
    return int(y * pl.display_size[1] / AI_H)


def draw_rectangle_lines(img, x1, y1, x2, y2, color, thickness=2):
    img.draw_line((x1, y1, x2, y1), color=color, thickness=thickness)
    img.draw_line((x2, y1, x2, y2), color=color, thickness=thickness)
    img.draw_line((x2, y2, x1, y2), color=color, thickness=thickness)
    img.draw_line((x1, y2, x1, y1), color=color, thickness=thickness)


def draw_screen(pl, gate, confirmer, box, digit, confidence, gap, diagnostic, confirmed):
    pl.osd_img.clear()
    rx1 = display_x(pl, ROI_X)
    ry1 = display_y(pl, ROI_Y)
    rx2 = display_x(pl, ROI_X + ROI_W)
    ry2 = display_y(pl, ROI_Y + ROI_H)
    draw_rectangle_lines(
        pl.osd_img, rx1, ry1, rx2, ry2, (255, 255, 255, 0), 3
    )
    pl.osd_img.draw_string_advanced(
        rx1, max(0, ry1 - 26), 20, "PAPER DIGIT", color=(255, 255, 255, 0)
    )

    if box is not None:
        x1, y1, x2, y2 = box
        draw_rectangle_lines(
            pl.osd_img,
            display_x(pl, ROI_X + x1),
            display_y(pl, ROI_Y + y1),
            display_x(pl, ROI_X + x2),
            display_y(pl, ROI_Y + y2),
            (255, 0, 255, 0),
            4,
        )

    pl.osd_img.draw_string_advanced(
        8,
        8,
        14,
        gate.status() + "  " + confirmer.status(),
        color=(255, 255, 255, 255),
    )
    pl.osd_img.draw_string_advanced(
        8, 28, 14, diagnostic, color=(255, 255, 255, 255)
    )

    panel_x = pl.display_size[0] - 170
    if digit is None:
        pl.osd_img.draw_string_advanced(
            panel_x, 18, 24, "NO DIGIT", color=(255, 255, 255, 255)
        )
    else:
        pl.osd_img.draw_string_advanced(
            panel_x, 14, 22, "DIGIT CNN", color=(255, 0, 255, 0)
        )
        pl.osd_img.draw_string_advanced(
            panel_x + 48, 42, 68, str(digit), color=(255, 0, 255, 0)
        )
        pl.osd_img.draw_string_advanced(
            panel_x,
            116,
            20,
            "P %.2f G %.2f" % (confidence, gap),
            color=(255, 255, 255, 255),
        )
    if confirmed is not None:
        pl.osd_img.draw_string_advanced(
            panel_x,
            154,
            26,
            "LOCKED %d" % confirmed,
            color=(255, 0, 255, 0),
        )


def packet_digit(value):
    try:
        digit = int(str(value).strip())
    except Exception:
        return None
    return digit if 1 <= digit <= 9 else None


def matching_ack(parts, digit):
    if len(parts) == 2 and str(parts[0]).lower() == "num":
        return packet_digit(parts[1]) == digit
    if len(parts) == 3 and str(parts[0]).lower() == "ack":
        return str(parts[1]).lower() == "num" and packet_digit(parts[2]) == digit
    return False


def uart_handshake(pl, link, digit):
    attempts = 0
    last_send = 0
    while True:
        now = time.ticks_ms()
        if attempts == 0 or time.ticks_diff(now, last_send) >= HANDSHAKE_RETRY_MS:
            if link.send_digit(digit):
                attempts += 1
                last_send = now
        for parts in link.read_packets():
            packet = "[" + ",".join(parts) + "]"
            print("RX:", packet)
            if matching_ack(parts, digit):
                print("RX VERIFIED:", packet)
                if link.send_confirmed(digit):
                    pl.osd_img.draw_string_advanced(
                        8,
                        205,
                        28,
                        "HANDSHAKE COMPLETE %d" % digit,
                        color=(255, 0, 255, 0),
                    )
                    pl.show_image()
                    time.sleep_ms(HANDSHAKE_DONE_DISPLAY_MS)
                    return True
        time.sleep_ms(HANDSHAKE_POLL_MS)


def main():
    os.stat(DIGIT_KMODEL)
    print("FILE OK:", DIGIT_KMODEL)
    link = UARTLink()
    pl = None
    gate = EmptySceneGate()
    confirmer = FourFrameConfirmer()
    recognizer = None
    frame_index = 0

    try:
        pl = PipeLine(
            rgb888p_size=[AI_W, AI_H],
            display_size=[DISPLAY_W, DISPLAY_H],
            display_mode=DISPLAY_MODE,
        )
        # Keep camera, LCD, IDE, and OSD coordinates on the same 640x480 canvas.
        pl.create(fps=30, to_ide=True)
        print("Actual display size:", pl.get_display_size())
        recognizer = DigitKModelRecognizer()

        print("================================================")
        print("K230 PAPER DIGIT -> DEDICATED CNN KMODEL -> UART3")
        print("OCR model: none")
        print("Template files: none")
        print("Empty learning frames:", WARMUP_FRAMES)
        print("CNN confidence threshold:", MIN_CNN_CONFIDENCE)
        print("CNN top1-top2 gap threshold:", MIN_CNN_GAP)
        print("Confirm: %d/%d equal frames" % (CONFIRM_VOTES, CONFIRM_FRAMES))
        print("================================================")

        while True:
            frame_index += 1
            frame = pl.get_frame()
            if frame is None:
                time.sleep_ms(10)
                continue

            roi = frame[:, ROI_Y : ROI_Y + ROI_H, ROI_X : ROI_X + ROI_W]
            roi_gray = roi[0] / 3.0 + roi[1] / 3.0 + roi[2] / 3.0
            scene_changed = gate.update(roi_gray)
            del roi
            del roi_gray

            box = None
            digit = None
            confidence = 0.0
            gap = 0.0
            diagnostic = "WAIT FOR PAPER DIGIT"

            if gate.ready and scene_changed:
                gray, ink, box, diagnostic = extract_digit_mask(frame)
                if box is not None:
                    normalized = make_model_image(ink, box)
                    digit, confidence, gap, raw_class = recognizer.run(normalized)
                    if digit is None:
                        diagnostic = "CNN REJECT C%d P%.2f G%.2f" % (
                            raw_class,
                            confidence,
                            gap,
                        )
                    else:
                        diagnostic = "DIGIT PRESENT -> CNN %d" % digit
                    del normalized
                del gray
                if ink is not None:
                    del ink

            confirmed = confirmer.update(digit)
            draw_screen(
                pl,
                gate,
                confirmer,
                box,
                digit,
                confidence,
                gap,
                diagnostic,
                confirmed,
            )
            pl.show_image()

            if frame_index % DEBUG_INTERVAL == 0:
                print(
                    "state=[%s] diag=[%s] digit=%s conf=%.4f gap=%.4f box=%s"
                    % (
                        gate.status(),
                        diagnostic,
                        str(digit),
                        confidence,
                        gap,
                        str(box),
                    )
                )

            if confirmed is not None:
                print("DIGIT CONFIRMED AND LOCKED:", confirmed)
                if uart_handshake(pl, link, confirmed):
                    break
            gc.collect()

    except KeyboardInterrupt:
        print("IDE interrupt")
    except Exception as exc:
        if str(exc) == "IDE interrupt":
            print("IDE interrupt")
        else:
            print("PROGRAM ERROR:", exc)
            raise
    finally:
        if recognizer is not None:
            try:
                recognizer.deinit()
            except Exception:
                pass
        if pl is not None:
            try:
                pl.destroy()
            except Exception:
                pass
        link.close()
        gc.collect()
        print("Program exited")


if __name__ == "__main__":
    main()
