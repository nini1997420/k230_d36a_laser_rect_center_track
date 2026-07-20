"""
K230 红线路口四槽位数字识别 v3

适用前提：
- 红线左侧最多 2 个黑色单数字，右侧最多 2 个。
- 数字实物大小和摆放间距固定，数字位于路口横线下方。
- 使用针对黑字、反光和阴影重训的 /sdcard/kmodel/digit_glare_int8.kmodel。

与 v2 的根本差异：
- 不再全图搜索无限个候选，只识别 L2/L1/R1/R2 四个物理槽位。
- 红线使用 7 个宽水平带快速取中心，不再扫描 70 条线。
- 每帧轮流处理 2 个槽位，最多只跑 2 次 KPU。
- 每个槽位先局部亮度归一化，再用近 3 帧的归一化笔画做多帧合并，
  减轻塑封膜反光将 5/6/8 笔画切断的问题。
"""

from libs.PipeLine import PipeLine
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import os
import time
import gc

from machine import FPIOA, UART


# ------------------------- 运行参数 -------------------------
AI_W = 320
AI_H = 240
DISPLAY_W = 640
DISPLAY_H = 480
DISPLAY_MODE = "lcd"
CAMERA_FPS = 30
TO_IDE = False                # LCD 仍显示；默认关闭 IDE JPEG 传输以提帧率
GC_EVERY_N_FRAMES = 45
DEBUG_EVERY_N_FRAMES = 30


# ------------------------- CNN -------------------------
DIGIT_KMODEL = "/sdcard/kmodel/digit_glare_int8.kmodel"
MODEL_SIZE = 32
MODEL_CONTENT_SIZE = 26
MIN_CNN_CONFIDENCE = 0.80
MIN_CNN_GAP = 0.25


# ------------------------- 红线几何 -------------------------
RED_MIN = 65.0
RED_DOMINANCE = 12.0
RED_RATIO = 1.12
RED_GAP_JOIN = 5

LINE_BAND_CENTERS = (226, 196, 166, 136, 106, 76, 46)
LINE_BAND_HALF_H = 11
LINE_COLUMN_VOTE_RATIO = 0.42
MIN_LINE_RUN_W = 3
MAX_LINE_RUN_W = 62
MAX_LINE_JUMP = 42
MIN_PATH_POINTS = 4
GEOMETRY_HOLD_FRAMES = 10
CROSS_MIN_RED_PIXELS = 82
CROSS_SEARCH_Y1 = 16
CROSS_SEARCH_Y2 = 155


# ------------------------- 四槽位几何 -------------------------
# 偏移量是 320x240 检测坐标，相对于十字路口处的红线中心。
# 依据实景图：左近/右近中心约为 -68/+50，左远/右远约为 -143/+127。
SLOT_SPECS = (
    ("L2", "L", -143),
    ("L1", "L", -68),
    ("R1", "R", 50),
    ("R2", "R", 127),
)
SLOT_HALF_W = 34
SLOT_Y_OFFSET_TOP = -5
SLOT_Y_OFFSET_BOTTOM = 205
SLOT_FALLBACK_Y1 = 54
SLOT_FALLBACK_Y2 = 232
MIN_SLOT_W = 25
MIN_SLOT_H = 65


# ------------------------- 槽内黑色笔画 -------------------------
# 局部阈值跟随每个槽位的平均亮度，阴影不再直接改变全图阈值。
LOCAL_DARK_RATIO = 0.78
LOCAL_DARK_MIN = 58.0
LOCAL_DARK_MAX = 145.0
LOCAL_NEUTRAL_DELTA = 62.0
MAX_ROW_FILL = 0.82
MAX_COL_FILL = 0.92
MIN_COL_PIXELS = 5
STRONG_COL_PIXELS = 10
COL_GAP_JOIN = 5
MIN_ROW_PIXELS = 2
ROW_GAP_JOIN = 4
FRAGMENT_GAP_JOIN = 48
MIN_FRAGMENT_H = 7
MIN_FRAGMENT_PIXELS = 18
MIN_GLYPH_W = 8
MAX_GLYPH_W = 62
MIN_GLYPH_H = 55
MAX_GLYPH_H = 150
MIN_GLYPH_FILL = 0.045
MAX_GLYPH_FILL = 0.68
MIN_GLYPH_HEIGHT_RATIO = 0.32
MAX_GLYPH_HEIGHT_RATIO = 0.88
MAX_GLYPH_CENTER_X_RATIO = 0.28
MIN_GLYPH_CENTER_Y_RATIO = 0.20
MAX_GLYPH_CENTER_Y_RATIO = 0.84
MIN_SLOT_CONTRAST = 24.0
MIN_SLOT_INK_RATIO = 0.008
MAX_SLOT_INK_RATIO = 0.42
GEOMETRY_STABLE_UPDATES = 2
GEOMETRY_CENTER_TOLERANCE = 11.0
GEOMETRY_SIZE_RATIO_TOLERANCE = 0.38

MASK_HISTORY = 3
MASK_UNION_MIN_VOTES = 2       # 3 次至少 2 次出现，不累积偶发阴影/地缝
MIN_MASK_HISTORY_FOR_CNN = 3
VOTE_WINDOW = 3
VOTE_REQUIRED = 2
RESULT_HOLD_UPDATES = 5

# 两侧数量必定相同：ONE=L1/R1，TWO=L2/L1/R1/R2。
MODE_OUTER_CONFIRM = 2
MODE_OUTER_HISTORY = 3
MODE_TWO_HOLD_FRAMES = 18


# ------------------------- 可选 UART -------------------------
UART_SEND_ENABLED = False
UART_TX_PIN = 32
UART_RX_PIN = 33
UART_ID = 3
UART_BAUD = 115200


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def find_runs_joined(flags, max_gap):
    runs = []
    start = -1
    last_true = -1
    for index in range(len(flags)):
        if bool(flags[index]):
            if start < 0:
                start = index
            last_true = index
        elif start >= 0 and index - last_true > max_gap:
            runs.append((start, last_true + 1))
            start = -1
            last_true = -1
    if start >= 0:
        runs.append((start, last_true + 1))
    return runs


class UARTEventLink:
    def __init__(self):
        self.uart = None
        if not UART_SEND_ENABLED:
            print("UART digit event disabled")
            return
        try:
            fpioa = FPIOA()
            fpioa.set_function(UART_TX_PIN, FPIOA.UART3_TXD, ie=0, oe=1)
            fpioa.set_function(UART_RX_PIN, FPIOA.UART3_RXD, ie=1, oe=0)
            uart_id = getattr(UART, "UART3", UART_ID)
            self.uart = UART(uart_id, baudrate=int(UART_BAUD))
        except Exception as exc:
            print("UART init failed:", exc)

    def send_digit(self, slot_name, side, digit):
        if self.uart is not None:
            packet = "[digit,%s,%s,%d]\r\n" % (slot_name, side, int(digit))
            self.uart.write(packet.encode())
            print("TX:", packet.strip())

    def close(self):
        if self.uart is not None:
            try:
                self.uart.deinit()
            except Exception:
                pass
            self.uart = None


def build_red_mask(frame_chw):
    r = frame_chw[0]
    g = frame_chw[1]
    b = frame_chw[2]
    return (
        (r >= RED_MIN)
        * (r >= g + RED_DOMINANCE)
        * (r >= b + RED_DOMINANCE)
        * (r >= g * RED_RATIO)
        * (r >= b * RED_RATIO)
    )


def fast_red_geometry(red_mask):
    """用 7 个宽带代替旧版 70 次逐行扫描。"""
    points = []
    previous_x = AI_W * 0.5
    acquired = False
    for center_y in LINE_BAND_CENTERS:
        y1 = max(0, center_y - LINE_BAND_HALF_H)
        y2 = min(AI_H, center_y + LINE_BAND_HALF_H + 1)
        height = y2 - y1
        column_votes = np.sum(red_mask[y1:y2, :], axis=0)
        required = max(3, int(height * LINE_COLUMN_VOTE_RATIO))
        runs = find_runs_joined(column_votes >= required, RED_GAP_JOIN)
        best = None
        best_distance = 100000.0
        for x1, x2 in runs:
            width = x2 - x1
            if width < MIN_LINE_RUN_W or width > MAX_LINE_RUN_W:
                continue
            center_x = (x1 + x2 - 1) * 0.5
            distance = abs(center_x - previous_x)
            limit = MAX_LINE_JUMP if acquired else AI_W * 0.48
            if distance <= limit and distance < best_distance:
                best = center_x
                best_distance = distance
        if best is not None:
            previous_x = best
            points.append((int(best + 0.5), center_y))
            acquired = True

    row_counts = np.sum(
        red_mask[CROSS_SEARCH_Y1:CROSS_SEARCH_Y2, :], axis=1
    )
    local_cross_y = int(np.argmax(row_counts))
    cross_strength = int(row_counts[local_cross_y])
    cross_y = local_cross_y + CROSS_SEARCH_Y1
    if cross_strength < CROSS_MIN_RED_PIXELS:
        cross_y = None
    # A wide diagonal cross tape can spread over many rows, so the global row
    # maximum may land at one far end. If it interrupts one internal path band,
    # that missing band is a better estimate of the actual main-line junction.
    present_y = []
    for point in points:
        present_y.append(point[1])
    for center_y in LINE_BAND_CENTERS[1:-1]:
        if center_y in present_y:
            continue
        has_below = False
        has_above = False
        for py in present_y:
            if py > center_y:
                has_below = True
            if py < center_y:
                has_above = True
        if has_below and has_above:
            cross_y = center_y
            break
    return points, cross_y, cross_strength


def path_x_at(points, y):
    if not points:
        return None
    best_x = points[0][0]
    best_dy = abs(points[0][1] - y)
    for x, py in points:
        dy = abs(py - y)
        if dy < best_dy:
            best_x = x
            best_dy = dy
    return best_x


class GeometryHold:
    def __init__(self):
        self.points = []
        self.cross_y = None
        self.age = 999

    def update(self, points, cross_y):
        if len(points) >= MIN_PATH_POINTS:
            if self.cross_y is not None and cross_y is not None:
                cross_y = int(self.cross_y * 0.65 + cross_y * 0.35 + 0.5)
            self.points = points
            if cross_y is not None:
                self.cross_y = cross_y
            self.age = 0
        else:
            self.age += 1
        if self.age > GEOMETRY_HOLD_FRAMES:
            self.points = []
            self.cross_y = None
        return self.points, self.cross_y


def slot_boxes(points, cross_y):
    if len(points) < MIN_PATH_POINTS:
        return []
    if cross_y is None:
        y1 = SLOT_FALLBACK_Y1
        y2 = SLOT_FALLBACK_Y2
        reference_y = (y1 + y2) // 2
    else:
        y1 = int(clamp(cross_y + SLOT_Y_OFFSET_TOP, 20, AI_H - MIN_SLOT_H))
        y2 = int(clamp(cross_y + SLOT_Y_OFFSET_BOTTOM, y1 + MIN_SLOT_H, AI_H - 3))
        reference_y = cross_y
    line_x = path_x_at(points, reference_y)
    if line_x is None:
        return []
    boxes = []
    for name, side, offset_x in SLOT_SPECS:
        cx = line_x + offset_x
        x1 = int(clamp(cx - SLOT_HALF_W, 3, AI_W - 3))
        x2 = int(clamp(cx + SLOT_HALF_W, 3, AI_W - 3))
        if x2 - x1 >= MIN_SLOT_W:
            boxes.append({
                "name": name, "side": side,
                "box": (x1, y1, x2, y2),
            })
    return boxes


def local_black_mask(frame_chw, box, red_mask):
    x1, y1, x2, y2 = box
    roi = frame_chw[:, y1:y2, x1:x2]
    r, g, b = roi[0], roi[1], roi[2]
    gray = r / 3.0 + g / 3.0 + b / 3.0
    local_mean = float(np.mean(gray))
    threshold = clamp(
        local_mean * LOCAL_DARK_RATIO, LOCAL_DARK_MIN, LOCAL_DARK_MAX
    )
    neutral = (
        (r <= g + LOCAL_NEUTRAL_DELTA) * (g <= r + LOCAL_NEUTRAL_DELTA)
        * (r <= b + LOCAL_NEUTRAL_DELTA) * (b <= r + LOCAL_NEUTRAL_DELTA)
        * (g <= b + LOCAL_NEUTRAL_DELTA) * (b <= g + LOCAL_NEUTRAL_DELTA)
    )
    ink = (
        (gray <= threshold)
        * neutral
        * (red_mask[y1:y2, x1:x2] == 0)
    )
    ink_pixels = float(np.sum(ink))
    ink_ratio = ink_pixels / max(1.0, float((y2 - y1) * (x2 - x1)))
    if ink_pixels > 0:
        ink_mean = float(np.sum(gray * ink)) / ink_pixels
        contrast = local_mean - ink_mean
    else:
        contrast = 0.0
    del gray
    del neutral
    return ink, threshold, contrast, ink_ratio


def best_glyph_box(ink):
    height = ink.shape[0]
    width = ink.shape[1]
    row_counts = np.sum(ink, axis=1)
    col_counts = np.sum(ink, axis=0)
    row_keep = (row_counts <= width * MAX_ROW_FILL).reshape((height, 1))
    col_keep = (col_counts <= height * MAX_COL_FILL).reshape((1, width))
    clean = ink * row_keep * col_keep
    clean_cols = np.sum(clean, axis=0)

    proposals = []
    for minimum in (STRONG_COL_PIXELS, MIN_COL_PIXELS):
        for gx1, gx2 in find_runs_joined(clean_cols >= minimum, COL_GAP_JOIN):
            if MIN_GLYPH_W <= gx2 - gx1 <= MAX_GLYPH_W:
                proposals.append((gx1, gx2, minimum))

    best = None
    best_score = -1.0
    for gx1, gx2, minimum in proposals:
        glyph_w = gx2 - gx1
        local_rows = np.sum(clean[:, gx1:gx2], axis=1)
        fragments = []
        for fy1, fy2 in find_runs_joined(
            local_rows >= MIN_ROW_PIXELS, ROW_GAP_JOIN
        ):
            pixels = int(np.sum(clean[fy1:fy2, gx1:gx2]))
            if fy2 - fy1 >= MIN_FRAGMENT_H and pixels >= MIN_FRAGMENT_PIXELS:
                fragments.append((fy1, fy2, pixels))
        if not fragments:
            continue

        gy1, gy2, total_pixels = fragments[0]
        for fy1, fy2, pixels in fragments[1:]:
            if fy1 - gy2 <= FRAGMENT_GAP_JOIN and fy2 - gy1 <= MAX_GLYPH_H:
                gy2 = fy2
                total_pixels += pixels
        glyph_h = gy2 - gy1
        if glyph_h < MIN_GLYPH_H or glyph_h > MAX_GLYPH_H:
            continue

        # 在选定高度内再次收紧 x，排除卡片斜边和阴影。
        refine_counts = np.sum(ink[gy1:gy2, gx1:gx2], axis=0)
        refine_runs = find_runs_joined(
            refine_counts >= minimum, COL_GAP_JOIN
        )
        strongest = None
        strongest_pixels = -1
        for rx1, rx2 in refine_runs:
            pixels = int(np.sum(refine_counts[rx1:rx2]))
            if rx2 - rx1 >= MIN_GLYPH_W and pixels > strongest_pixels:
                strongest = (rx1, rx2)
                strongest_pixels = pixels
        if strongest is None:
            continue
        bx1 = gx1 + strongest[0]
        bx2 = gx1 + strongest[1]
        glyph_w = bx2 - bx1
        glyph_center_x = (bx1 + bx2) * 0.5
        glyph_center_y = (gy1 + gy2) * 0.5
        height_ratio = glyph_h / max(1.0, float(height))
        center_y_ratio = glyph_center_y / max(1.0, float(height))
        if (
            height_ratio < MIN_GLYPH_HEIGHT_RATIO
            or height_ratio > MAX_GLYPH_HEIGHT_RATIO
            or center_y_ratio < MIN_GLYPH_CENTER_Y_RATIO
            or center_y_ratio > MAX_GLYPH_CENTER_Y_RATIO
        ):
            continue
        if (
            abs(glyph_center_x - width * 0.5)
            > width * MAX_GLYPH_CENTER_X_RATIO
        ):
            # Fixed slot: a line close to the card edge is border/shadow, not digit 1.
            continue
        pixels = int(np.sum(ink[gy1:gy2, bx1:bx2]))
        fill = pixels / max(1.0, float(glyph_w * glyph_h))
        if fill < MIN_GLYPH_FILL or fill > MAX_GLYPH_FILL:
            continue
        score = pixels + glyph_h * 2.0 + fill * 60.0
        if score > best_score:
            best_score = score
            best = (bx1, gy1, bx2, gy2)
    return best


def normalize_glyph(ink, box):
    x1, y1, x2, y2 = box
    src_w = x2 - x1
    src_h = y2 - y1
    scale = min(MODEL_CONTENT_SIZE / src_w, MODEL_CONTENT_SIZE / src_h)
    dst_w = max(1, int(src_w * scale + 0.5))
    dst_h = max(1, int(src_h * scale + 0.5))
    offset_x = (MODEL_SIZE - dst_w) // 2
    offset_y = (MODEL_SIZE - dst_h) // 2
    output = np.zeros((1, 1, MODEL_SIZE, MODEL_SIZE), dtype=np.uint8)
    for dy in range(dst_h):
        sy = y1 + min(src_h - 1, int(dy * src_h / dst_h))
        for dx in range(dst_w):
            sx = x1 + min(src_w - 1, int(dx * src_w / dst_w))
            if ink[sy, sx]:
                output[0, 0, offset_y + dy, offset_x + dx] = 255
    return output


def merge_normalized_masks(history):
    votes = np.zeros((1, 1, MODEL_SIZE, MODEL_SIZE), dtype=np.uint8)
    for mask in history:
        votes = votes + (mask > 0)
    merged = np.zeros((1, 1, MODEL_SIZE, MODEL_SIZE), dtype=np.uint8)
    # Explicit assignment keeps the KModel input dtype uint8 on ulab.
    for y in range(MODEL_SIZE):
        for x in range(MODEL_SIZE):
            if votes[0, 0, y, x] >= MASK_UNION_MIN_VOTES:
                merged[0, 0, y, x] = 255
    del votes
    return merged


class DigitKModelRecognizer:
    def __init__(self):
        self.kpu = nn.kpu()
        self.kpu.load_kmodel(DIGIT_KMODEL)
        print("Digit CNN ready:", DIGIT_KMODEL)

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
        for index in range(10):
            if index != best_class:
                value = float(probabilities[index])
                if value > second_probability:
                    second_probability = value
        gap = best_probability - second_probability
        del input_tensor
        del output_tensor
        del logits
        del exp_values
        del probabilities
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


class SlotState:
    def __init__(self, name, side, link):
        self.name = name
        self.side = side
        self.link = link
        self.mask_history = []
        self.vote_history = []
        self.confirmed = None
        self.confidence = 0.0
        self.hold = 0
        self.sent = None
        self.roi_box = None
        self.glyph_box = None
        self.threshold = 0.0
        self.contrast = 0.0
        self.ink_ratio = 0.0
        self.last_geometry = None
        self.stable_updates = 0
        self.miss_count = 0

    def reset(self):
        self.mask_history = []
        self.vote_history = []
        self.confirmed = None
        self.confidence = 0.0
        self.hold = 0
        self.sent = None
        self.glyph_box = None
        self.last_geometry = None
        self.stable_updates = 0
        self.miss_count = 0

    def vote(self):
        winner = None
        best_count = 0
        for digit in range(1, 10):
            count = 0
            for value in self.vote_history:
                if value == digit:
                    count += 1
            if count > best_count:
                winner = digit
                best_count = count
        return winner if best_count >= VOTE_REQUIRED else None

    def update_result(self, digit, confidence):
        self.vote_history.append(-1 if digit is None else int(digit))
        if len(self.vote_history) > VOTE_WINDOW:
            self.vote_history.pop(0)
        winner = self.vote()
        if winner is not None:
            self.confirmed = winner
            self.confidence = confidence
            self.hold = RESULT_HOLD_UPDATES
            if self.sent != winner:
                print("CONFIRMED %s %s:%d" % (self.name, self.side, winner))
                self.link.send_digit(self.name, self.side, winner)
                self.sent = winner
        elif self.hold > 0:
            self.hold -= 1
        else:
            self.confirmed = None
            if digit is None:
                self.sent = None

    def geometry_is_stable(self, local_box):
        x1, y1, x2, y2 = local_box
        current = (
            (x1 + x2) * 0.5, (y1 + y2) * 0.5,
            x2 - x1, y2 - y1,
        )
        stable = False
        if self.last_geometry is not None:
            old_cx, old_cy, old_w, old_h = self.last_geometry
            center_delta = (
                (current[0] - old_cx) ** 2
                + (current[1] - old_cy) ** 2
            ) ** 0.5
            width_ratio = abs(current[2] - old_w) / max(1.0, float(old_w))
            height_ratio = abs(current[3] - old_h) / max(1.0, float(old_h))
            stable = (
                center_delta <= GEOMETRY_CENTER_TOLERANCE
                and width_ratio <= GEOMETRY_SIZE_RATIO_TOLERANCE
                and height_ratio <= GEOMETRY_SIZE_RATIO_TOLERANCE
            )
        if stable:
            self.stable_updates += 1
        else:
            self.stable_updates = 1
            self.mask_history = []
        self.last_geometry = current
        return self.stable_updates >= GEOMETRY_STABLE_UPDATES

    def miss(self):
        self.miss_count += 1
        if self.miss_count >= 2:
            self.mask_history = []
            self.glyph_box = None
            self.last_geometry = None
            self.stable_updates = 0
        self.update_result(None, 0.0)


class CountMode:
    def __init__(self):
        self.mode = "ONE"
        self.outer_history = []
        self.two_hold = 0

    def update(self, observations):
        outer_pair = (
            observations.get("L2") is not None
            and observations.get("R2") is not None
            and observations["L2"]["geometry_ready"]
            and observations["R2"]["geometry_ready"]
        )
        inner_pair = (
            observations.get("L1") is not None
            and observations.get("R1") is not None
            and observations["L1"]["geometry_ready"]
            and observations["R1"]["geometry_ready"]
        )
        four_present = outer_pair and inner_pair
        self.outer_history.append(1 if four_present else 0)
        if len(self.outer_history) > MODE_OUTER_HISTORY:
            self.outer_history.pop(0)
        outer_votes = 0
        for value in self.outer_history:
            outer_votes += value
        old_mode = self.mode
        if outer_votes >= MODE_OUTER_CONFIRM:
            self.mode = "TWO"
            self.two_hold = MODE_TWO_HOLD_FRAMES
        elif self.mode == "TWO":
            if self.two_hold > 0:
                self.two_hold -= 1
            elif inner_pair:
                self.mode = "ONE"
        return old_mode != self.mode


def observe_slot(frame, red_mask, slot, state):
    state.roi_box = slot["box"]
    ink, threshold, contrast, ink_ratio = local_black_mask(
        frame, slot["box"], red_mask
    )
    state.threshold = threshold
    state.contrast = contrast
    state.ink_ratio = ink_ratio
    if (
        contrast < MIN_SLOT_CONTRAST
        or ink_ratio < MIN_SLOT_INK_RATIO
        or ink_ratio > MAX_SLOT_INK_RATIO
    ):
        del ink
        state.miss()
        return None
    local_box = best_glyph_box(ink)
    if local_box is None:
        del ink
        state.miss()
        return None
    state.miss_count = 0
    geometry_ready = state.geometry_is_stable(local_box)
    x1, y1, x2, y2 = slot["box"]
    state.glyph_box = (
        x1 + local_box[0], y1 + local_box[1],
        x1 + local_box[2], y1 + local_box[3],
    )
    normalized = normalize_glyph(ink, local_box)
    del ink
    state.mask_history.append(normalized)
    if len(state.mask_history) > MASK_HISTORY:
        state.mask_history.pop(0)
    return {
        "local_box": local_box,
        "geometry_ready": geometry_ready,
    }


def process_observation(observation, state, recognizer):
    geometry_ready = observation["geometry_ready"]
    if (
        not geometry_ready
        or len(state.mask_history) < MIN_MASK_HISTORY_FOR_CNN
    ):
        return
    merged = merge_normalized_masks(state.mask_history)
    digit, confidence, gap, raw_class = recognizer.run(merged)
    del merged
    state.update_result(digit, confidence)


def display_x(pl, x):
    return int(x * pl.display_size[0] / AI_W)


def display_y(pl, y):
    return int(y * pl.display_size[1] / AI_H)


def draw_rectangle(img, pl, box, color, thickness):
    x1, y1, x2, y2 = box
    x1, y1 = display_x(pl, x1), display_y(pl, y1)
    x2, y2 = display_x(pl, x2), display_y(pl, y2)
    img.draw_line((x1, y1, x2, y1), color=color, thickness=thickness)
    img.draw_line((x2, y1, x2, y2), color=color, thickness=thickness)
    img.draw_line((x2, y2, x1, y2), color=color, thickness=thickness)
    img.draw_line((x1, y2, x1, y1), color=color, thickness=thickness)


def draw_screen(pl, points, cross_y, states, fps, detect_ms, processed_names):
    pl.osd_img.clear()
    for index in range(1, len(points)):
        x1, y1 = points[index - 1]
        x2, y2 = points[index]
        pl.osd_img.draw_line(
            (display_x(pl, x1), display_y(pl, y1),
             display_x(pl, x2), display_y(pl, y2)),
            color=(255, 0, 255, 0), thickness=3
        )
    if cross_y is not None:
        cx = path_x_at(points, cross_y)
        if cx is not None:
            pl.osd_img.draw_cross(
                display_x(pl, cx), display_y(pl, cross_y),
                color=(255, 255, 0, 255), size=8, thickness=2
            )

    for state in states.values():
        if state.roi_box is not None:
            draw_rectangle(pl.osd_img, pl, state.roi_box, (255, 100, 100, 100), 1)
            x1, y1, _, _ = state.roi_box
            pl.osd_img.draw_string_advanced(
                display_x(pl, x1), display_y(pl, y1), 14,
                state.name, color=(255, 180, 180, 180)
            )
        if state.confirmed is not None and state.glyph_box is not None:
            color = (255, 255, 255, 0) if state.side == "L" else (255, 0, 255, 255)
            draw_rectangle(pl.osd_img, pl, state.glyph_box, color, 4)
            gx1, gy1, _, _ = state.glyph_box
            pl.osd_img.draw_string_advanced(
                display_x(pl, gx1), max(0, display_y(pl, gy1) - 24), 20,
                "%s:%d P%.2f" % (state.side, state.confirmed, state.confidence),
                color=color
            )

    pl.osd_img.draw_string_advanced(
        5, 4, 16,
        "FPS %.1f DET %.0fms PATH %d CROSS %s RUN %s" %
        (fps, detect_ms, len(points), str(cross_y), processed_names),
        color=(255, 255, 255, 255)
    )


def main():
    os.stat(DIGIT_KMODEL)
    print("FILE OK:", DIGIT_KMODEL)
    pl = None
    recognizer = None
    link = UARTEventLink()
    geometry = GeometryHold()
    states = {}
    for name, side, offset in SLOT_SPECS:
        states[name] = SlotState(name, side, link)
    count_mode = CountMode()
    two_pair_cursor = 0
    frame_index = 0
    fps = 0.0
    fps_count = 0
    fps_start = time.ticks_ms()
    detect_ms = 0.0
    points = []
    cross_y = None
    processed_names = "-"

    try:
        pl = PipeLine(
            rgb888p_size=[AI_W, AI_H],
            display_size=[DISPLAY_W, DISPLAY_H],
            display_mode=DISPLAY_MODE,
        )
        pl.create(fps=CAMERA_FPS, to_ide=TO_IDE)
        print("Actual display size:", pl.get_display_size())
        recognizer = DigitKModelRecognizer()
        print("RED-LINE FOUR-SLOT DIGIT v3 READY")

        while True:
            frame = pl.get_frame()
            if frame is None:
                time.sleep_ms(3)
                continue
            frame_index += 1
            fps_count += 1
            start_ms = time.ticks_ms()
            red_mask = build_red_mask(frame)
            fresh_points, fresh_cross_y, cross_strength = fast_red_geometry(red_mask)
            points, cross_y = geometry.update(fresh_points, fresh_cross_y)
            slots = slot_boxes(points, cross_y)
            slot_map = {}
            for slot in slots:
                slot_map[slot["name"]] = slot
                states[slot["name"]].roi_box = slot["box"]
            for name, state in states.items():
                if name not in slot_map:
                    state.roi_box = None
                    state.miss()

            observations = {}
            for name, slot in slot_map.items():
                observations[name] = observe_slot(
                    frame, red_mask, slot, states[name]
                )

            mode_changed = count_mode.update(observations)
            if mode_changed and count_mode.mode == "ONE":
                states["L2"].reset()
                states["R2"].reset()

            if count_mode.mode == "ONE":
                left_name = (
                    "L1" if observations.get("L1") is not None
                    else "L2"
                )
                right_name = (
                    "R1" if observations.get("R1") is not None
                    else "R2"
                )
                active_names = (left_name, right_name)
            else:
                if two_pair_cursor % 2 == 0:
                    active_names = ("L2", "L1")
                else:
                    active_names = ("R1", "R2")
                two_pair_cursor += 1

            processed = []
            for name in active_names:
                observation = observations.get(name)
                if observation is not None:
                    process_observation(
                        observation, states[name], recognizer
                    )
                    processed.append(name)
            processed_names = count_mode.mode + ":" + (
                "/".join(processed) if processed else "-"
            )
            detect_ms = float(time.ticks_diff(time.ticks_ms(), start_ms))
            del red_mask

            now = time.ticks_ms()
            elapsed = time.ticks_diff(now, fps_start)
            if elapsed >= 1000:
                fps = fps_count * 1000.0 / elapsed
                fps_count = 0
                fps_start = now

            draw_screen(
                pl, points, cross_y, states, fps, detect_ms, processed_names
            )
            pl.show_image()

            if frame_index % DEBUG_EVERY_N_FRAMES == 0:
                results = []
                thresholds = []
                for name, state in states.items():
                    if state.confirmed is not None:
                        results.append("%s:%d" % (name, state.confirmed))
                    thresholds.append(
                        "%s=T%.0f/C%.0f/I%.2f"
                        % (name, state.threshold, state.contrast, state.ink_ratio)
                    )
                print(
                    "fps=%.1f det=%.0fms path=%d cross=%s run=%s results=%s thr=%s"
                    % (fps, detect_ms, len(points), str(cross_y), processed_names,
                       str(results), "/".join(thresholds))
                )
            if frame_index % GC_EVERY_N_FRAMES == 0:
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
