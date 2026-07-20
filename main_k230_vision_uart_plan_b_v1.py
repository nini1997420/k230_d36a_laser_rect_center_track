# -*- coding: utf-8 -*-
"""
方案 B / K230 视觉端第一版。

职责：
    1. 识别矩形中心和激光点中心。
    2. 保留目标锁定、重捕获和坐标级滤波。
    3. 通过 UART3 把最新视觉观测发送给 MSPM0。
    4. 在 ST7701 屏幕上低频显示识别结果，不占用 IDE 图传。

本程序不初始化 D36A，不输出 STEP/DIR，不运行 PD、Ramp、虚拟位置或软件限位。

唯一高频串口帧为下位机 app_aim_protocol.c 实际接收的22字节二进制帧：
    AA 55 01 01 | sequence(u16 LE) | capture_ms(u32 LE)
    | rect_x(u16 LE) | rect_y(u16 LE) | laser_x(u16 LE) | laser_y(u16 LE)
    | valid_flags(u8) | tracking_state(u8) | crc16(u16 LE)

字段说明：
    坐标使用 640x480 逻辑坐标；无效坐标为 65535。
    valid_flags bit0=矩形坐标可用，bit1=激光坐标可用，
                bit2=目标均已锁定，bit3=测量仍在新鲜时限内；bit4~7固定为0。
本文件包含全部运行依赖，上传这一个 .py 文件即可。
"""

import gc
import math
import os
import sys
import time

import cv_lite
from machine import FPIOA, UART, Pin
from media.sensor import *
from media.display import *
from media.media import *


# -----------------------------------------------------------------------------
# 运行配置
# -----------------------------------------------------------------------------

SENSOR_ID = 2

# 这里是模式匹配请求；Yahboom CanMV v1.4.3实机会适配为GC2093 1280x960@90，
# 两个输出通道仍分别缩放为320x240和160x120。实测纯snapshot约87.15 FPS。
SENSOR_INPUT_WIDTH = 640
SENSOR_INPUT_HEIGHT = 480
SENSOR_FPS = 90

DETECT_WIDTH = 320
DETECT_HEIGHT = 240
LASER_DETECT_WIDTH = 160
LASER_DETECT_HEIGHT = 120
LOGICAL_WIDTH = 640
LOGICAL_HEIGHT = 480

RECT_CHANNEL = CAM_CHN_ID_0
LASER_CHANNEL = CAM_CHN_ID_1

# 第一版优先识别可靠性：矩形用 RGB888，激光用独立 RGB565 通道。
# 两个通道由同一传感器并行输出，但主循环仍分别取得各自图像。

ENABLE_LCD = True
LCD_REFRESH_EVERY_N_RECT_FRAMES = 5  # 每5次矩形识别刷新一次，约4~5 Hz
LCD_TO_IDE = False                    # 禁止IDE图传抢带宽/算力
LCD_QUALITY = 35

ENABLE_UART = True
UART_ID = 3
UART_TX_PIN = 32
UART_RX_PIN = 33
UART_BAUD = 460800                    # MSPM0 UART0必须同步配置为460800 8N1

# 22字节二进制帧在460800 baud下可显著缩短阻塞发送时间。双通道取帧会产生
# 长短循环，不能使用毫秒门限，否则短循环会被主动漏发，导致TX低于FPS。
PLOT_INTERVAL_MS = 0
GC_CHECK_INTERVAL_FRAMES = 120
GC_FREE_THRESHOLD = 180000

# 四时隙协作调度：1帧矩形 + 3帧激光。每循环只跑一种重识别，避免耗时叠加。
# 传感器实测87 FPS时：数据包约87 Hz、激光真实检测约65 Hz、矩形约22 Hz。
RECT_SLOT_PERIOD = 4

INVALID_COORD = 65535
RECT_FRESH_MAX_AGE_MS = 70
LASER_FRESH_MAX_AGE_MS = 35
HOLD_MAX_AGE_MS = 220

# MSPM0 App/app_aim_protocol.h 当前实际协议（不是README中的旧24字节草案）。
AIM_FRAME_SIZE = 22
AIM_CRC_DATA_SIZE = 20
AIM_HEADER0 = 0xAA
AIM_HEADER1 = 0x55
AIM_PROTOCOL_VERSION = 0x01
AIM_MSG_VISION_OBSERVATION = 0x01

LASER_IO_PIN = 35
LASER_ACTIVE_LEVEL = 1
LASER_AUTO_ON = True

DETECT_TO_LOGICAL_X = LOGICAL_WIDTH / float(DETECT_WIDTH)
DETECT_TO_LOGICAL_Y = LOGICAL_HEIGHT / float(DETECT_HEIGHT)
LASER_TO_LOGICAL_X = LOGICAL_WIDTH / float(LASER_DETECT_WIDTH)
LASER_TO_LOGICAL_Y = LOGICAL_HEIGHT / float(LASER_DETECT_HEIGHT)
IMAGE_SHAPE = [DETECT_HEIGHT, DETECT_WIDTH]


def ticks_diff_ms(now, old):
    return time.ticks_diff(now, old)


def perf_ms(start_us):
    return time.ticks_diff(time.ticks_us(), start_us) / 1000.0


def make_crc16_ccitt_false_table():
    table = []
    for value in range(256):
        crc = value << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


CRC16_CCITT_FALSE_TABLE = make_crc16_ccitt_false_table()


def crc16_ccitt_false(data, length):
    crc = 0xFFFF
    for index in range(int(length)):
        crc = ((crc << 8) ^ CRC16_CCITT_FALSE_TABLE[((crc >> 8) ^ data[index]) & 0xFF]) & 0xFFFF
    return crc


def write_le16(buffer, offset, value):
    value = int(value) & 0xFFFF
    buffer[offset] = value & 0xFF
    buffer[offset + 1] = (value >> 8) & 0xFF


def write_le32(buffer, offset, value):
    value = int(value) & 0xFFFFFFFF
    buffer[offset] = value & 0xFF
    buffer[offset + 1] = (value >> 8) & 0xFF
    buffer[offset + 2] = (value >> 16) & 0xFF
    buffer[offset + 3] = (value >> 24) & 0xFF


def clamp_int(value, low, high):
    value = int(round(value))
    if value < low:
        return low
    if value > high:
        return high
    return value


def coord_or_invalid(position, index, fresh):
    if not fresh or position is None:
        return INVALID_COORD
    limit = LOGICAL_WIDTH - 1 if index == 0 else LOGICAL_HEIGHT - 1
    return clamp_int(position[index], 0, limit)


UART_ECHO_TELEMETRY = False

RECT_CFG = {
    "min_area_detect": 120.0,
    "max_area_ratio": 0.46,
    "min_w_detect": 12,
    "min_h_detect": 9,
    "aspect_min": 1.05,
    "aspect_max": 2.38,
    "target_aspect": 1.50,
    "quality_high": 0.36,
    "quality_low": 0.08,
    "quality_recover": 0.22,
    "max_candidates": 16,

    # v10.11：强目标身份锁定。v10数据中的乱抖主要来自矩形候选跳到远处干扰框，
    # 因此锁定后不允许直接接受远离上一帧轨迹的全局最高分候选。
    "gate_track_px": 86.0,
    "gate_recover_px": 190.0,
    "far_recover_delay_ms": 420,
    "reacquire_stable_px": 36.0,
    "reacquire_confirm_frames": 3,
    "reset_after_lost": True,
    "cv_canny_low": 22,
    "cv_canny_high": 84,
    "cv_approx_epsilon": 0.023,
    "cv_area_min_ratio": 0.0010,
    "cv_max_angle_cos": 0.46,
    "cv_gaussian_blur_size": 3,
    "jump_guard_px": 100.0,
    "acquire_frames": 2,
    "short_hold_ms": 540,
    "lost_ms": 1300,
}

LASER_CFG = {
    "core_l_min": 18,
    "core_l_max": 88,
    "core_a_min": 30,
    "core_a_max": 127,
    "core_b_min": 5,
    "core_b_max": 127,
    "halo_l_min": 12,
    "halo_l_max": 100,
    "halo_a_min": 16,
    "halo_a_max": 127,
    "halo_b_min": -8,
    "halo_b_max": 127,
    "pixels_min": 2,
    "area_min": 2,
    "area_max": 220,
    "target_area": 10.0,
    "max_w_det": 24.0,
    "max_h_det": 24.0,
    "max_aspect": 3.2,
    "min_density": 0.12,
    "merge_margin": 0,
    "gate_locked_px_det": 28.0,
    "gate_reacquire_px_det": 150.0,
    "acquire_jump_px_det": 34.0,
    "acquire_frames": 2,
    "max_acquire_candidates": 8,
    "min_acquire_score": 0.42,
    "min_locked_score": 0.20,
    "ambiguity_margin": 0.06,
    "position_alpha": 0.68,
    "velocity_alpha": 0.18,
    "velocity_limit_logical": 1000.0,
    "anchor_keep_ms": 2800,
    "anchor_gate_det": 42.0,
    "dynamic_gate_max_det": 56.0,
    "rect_roi_margin_det": 58.0,
    "full_scan_fallback_ms": 1500,
    "full_scan_period": 6,
}

# 激光通道从320x240降为160x120后，所有检测空间参数按比例缩放。
# 输出坐标仍统一换算到640x480，因此下位机协议无需改变。
LASER_CFG["pixels_min"] = 1
LASER_CFG["area_min"] = 1
LASER_CFG["area_max"] = 80
LASER_CFG["target_area"] = 3.0
LASER_CFG["max_w_det"] = 14.0
LASER_CFG["max_h_det"] = 14.0
LASER_CFG["gate_locked_px_det"] = 14.0
LASER_CFG["gate_reacquire_px_det"] = 75.0
LASER_CFG["acquire_jump_px_det"] = 17.0
LASER_CFG["anchor_gate_det"] = 21.0
LASER_CFG["dynamic_gate_max_det"] = 28.0
LASER_CFG["rect_roi_margin_det"] = 6.0

def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def sign_of(value):
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def move_toward(current, target, max_delta):
    if current < target:
        return min(target, current + max_delta)
    if current > target:
        return max(target, current - max_delta)
    return current


def ticks_diff_ms(now, old):
    return time.ticks_diff(now, old)


def perf_ms(start_us):
    return time.ticks_diff(time.ticks_us(), start_us) / 1000.0


def detect_to_logical_x(x):
    return float(x) * DETECT_TO_LOGICAL_X


def detect_to_logical_y(y):
    return float(y) * DETECT_TO_LOGICAL_Y


def logical_to_detect_x(x):
    return float(x) / DETECT_TO_LOGICAL_X


def logical_to_detect_y(y):
    return float(y) / DETECT_TO_LOGICAL_Y


def box_center(box):
    x, y, w, h = box
    return (float(x) + 0.5 * float(w), float(y) + 0.5 * float(h))


def logical_center_from_det(cx, cy):
    return (detect_to_logical_x(cx), detect_to_logical_y(cy))


def laser_center_to_logical(cx, cy):
    return (float(cx) * LASER_TO_LOGICAL_X, float(cy) * LASER_TO_LOGICAL_Y)


def rect_box_to_laser(box):
    if box is None:
        return None
    x, y, w, h = box
    sx = LASER_DETECT_WIDTH / float(DETECT_WIDTH)
    sy = LASER_DETECT_HEIGHT / float(DETECT_HEIGHT)
    return (float(x) * sx, float(y) * sy, float(w) * sx, float(h) * sy)


def dist2(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy

# ============================================================
# 6. GPIO / PWM / UART
# ============================================================
def write_pin(pin, value):
    level = 1 if value else 0
    try:
        pin.value(level)
    except Exception:
        if level and hasattr(pin, "on"):
            pin.on()
        elif (not level) and hasattr(pin, "off"):
            pin.off()
        else:
            raise


def make_gpio_output(pin_number, initial_value=0):
    pin_number = int(pin_number)
    fpioa = FPIOA()
    func = getattr(FPIOA, "GPIO%d" % pin_number, None)
    if func is not None:
        try:
            fpioa.set_function(pin_number, func, ie=0, oe=1, pu=0, pd=0)
        except TypeError:
            fpioa.set_function(pin_number, func)
        except Exception as e:
            print("FPIOA GPIO%d mapping warning:" % pin_number, e)
    pull_none = getattr(Pin, "PULL_NONE", None)
    try:
        if pull_none is None:
            pin = Pin(pin_number, Pin.OUT, drive=7)
        else:
            pin = Pin(pin_number, Pin.OUT, pull=pull_none, drive=7)
    except TypeError:
        try:
            pin = Pin(pin_number, Pin.OUT)
        except TypeError:
            pin = Pin(pin_number)
    write_pin(pin, initial_value)
    return pin


class UARTLink:
    def __init__(self):
        self.uart = None
        self.buffer = ""
        self.tx_packets = 0
        self.rx_packets = 0
        if not ENABLE_UART:
            print("External UART disabled")
            return
        try:
            fpioa = FPIOA()
            fpioa.set_function(UART_TX_PIN, FPIOA.UART3_TXD, ie=0, oe=1)
            fpioa.set_function(UART_RX_PIN, FPIOA.UART3_RXD, ie=1, oe=0)
            uart_id = getattr(UART, "UART3", UART_ID)
            try:
                self.uart = UART(uart_id, baudrate=int(UART_BAUD), bits=8, parity=None, stop=1, timeout=0)
            except TypeError:
                self.uart = UART(uart_id, baudrate=int(UART_BAUD))
        except Exception as e:
            print("UART3 init failed; continue without external UART:", e)
            self.uart = None

    def is_ready(self):
        return self.uart is not None

    def send(self, text):
        message = str(text)
        is_telemetry = message.startswith("[plot,") or message.startswith("[diag,")
        if UART_ECHO_TELEMETRY or not is_telemetry:
            print(message)
        if self.uart is None:
            return False
        try:
            self.uart.write((message + "\r\n").encode("utf-8"))
            self.tx_packets += 1
            return True
        except Exception as e:
            print("UART write failed:", e)
            return False

    def send_binary(self, frame):
        if self.uart is None:
            return False
        try:
            self.uart.write(frame)
            self.tx_packets += 1
            return True
        except Exception as e:
            print("UART binary write failed:", e)
            return False

    def read_packets(self, max_packets=6):
        packets = []
        if self.uart is None:
            return packets
        try:
            if hasattr(self.uart, "any") and self.uart.any() <= 0:
                return packets
            data = self.uart.read()
        except Exception:
            data = None
        if not data:
            return packets
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except Exception:
                data = ""
        self.buffer += str(data)
        if len(self.buffer) > 2048:
            s = self.buffer.rfind("[")
            self.buffer = self.buffer[s:] if s >= 0 else ""
        while len(packets) < int(max_packets) and "[" in self.buffer and "]" in self.buffer:
            s = self.buffer.find("[")
            e = self.buffer.find("]", s)
            if e <= s:
                break
            content = self.buffer[s + 1:e]
            self.buffer = self.buffer[e + 1:]
            parts = [p.strip() for p in content.split(",")]
            if parts and parts[0]:
                packets.append(parts)
                self.rx_packets += 1
        return packets

    def close(self):
        if self.uart is not None:
            try:
                self.uart.deinit()
            except Exception:
                pass


class LaserTTL35:
    def __init__(self, pin_no=LASER_IO_PIN, active_level=LASER_ACTIVE_LEVEL):
        self.pin_no = int(pin_no)
        self.active_level = 1 if active_level else 0
        self.pin = make_gpio_output(self.pin_no, 1 - self.active_level)
        self.enabled = False
        self.off()

    def set(self, enabled):
        self.enabled = bool(enabled)
        level = self.active_level if self.enabled else (1 - self.active_level)
        write_pin(self.pin, level)
        return self.enabled

    def on(self):
        return self.set(True)

    def off(self):
        return self.set(False)

    def set_active_level(self, level):
        was = bool(self.enabled)
        self.active_level = 1 if float(level) >= 0.5 else 0
        self.set(was)
        return self.active_level

    def deinit(self):
        try:
            self.off()
        except Exception:
            pass


class RectTrackerLite:
    def __init__(self):
        self.locked = False
        self.center = None
        self.raw_center = None
        self.velocity = (0.0, 0.0)
        self.box_det = None
        self.corners_det = None
        self.last_ms = time.ticks_ms()
        self.acquire_count = 0
        self.miss_frames = 0
        self.candidate_count = 0
        self.confidence = 0.0
        self.state = 0
        self.pending_det = None
        self.pending_count = 0
        self.reject_far_count = 0

    def reset(self):
        self.locked = False
        self.center = None
        self.raw_center = None
        self.velocity = (0.0, 0.0)
        self.box_det = None
        self.corners_det = None
        self.acquire_count = 0
        self.miss_frames = 0
        self.confidence = 0.0
        self.state = 0
        self.pending_det = None
        self.pending_count = 0
        self.reject_far_count = 0

    def _candidate_quality(self, r):
        x, y, w, h = float(r[0]), float(r[1]), float(r[2]), float(r[3])
        if w < RECT_CFG["min_w_detect"] or h < RECT_CFG["min_h_detect"]:
            return None
        area = w * h
        if area < RECT_CFG["min_area_detect"] or area > RECT_CFG["max_area_ratio"] * DETECT_WIDTH * DETECT_HEIGHT:
            return None
        aspect = max(w / max(1.0, h), h / max(1.0, w))
        normal_aspect = w / max(1.0, h)
        if normal_aspect < RECT_CFG["aspect_min"] or normal_aspect > RECT_CFG["aspect_max"]:
            return None
        cx, cy = x + 0.5 * w, y + 0.5 * h
        aspect_score = clamp(1.0 - abs(normal_aspect - RECT_CFG["target_aspect"]) / 0.95, 0.0, 1.0)
        area_score = clamp(area / 5500.0, 0.0, 1.0)
        center_score = 0.0
        if self.center is not None:
            last_det = (logical_to_detect_x(self.center[0]), logical_to_detect_y(self.center[1]))
            d = math.sqrt((cx - last_det[0]) ** 2 + (cy - last_det[1]) ** 2)
            gate = RECT_CFG["gate_track_px"] if self.locked else RECT_CFG["gate_recover_px"]
            center_score = clamp(1.0 - d / max(1.0, gate), 0.0, 1.0)
        else:
            center_score = 0.25
        q = 0.46 * aspect_score + 0.28 * area_score + 0.26 * center_score
        return {"box_det": (x, y, w, h), "center_det": (cx, cy), "area": area, "quality": q, "corners_det": tuple((float(r[i]), float(r[i+1])) for i in range(4, min(len(r), 12), 2))}

    def _detect_candidates(self, img_np):
        candidates = []
        try:
            rects = cv_lite.rgb888_find_rectangles_with_corners(
                IMAGE_SHAPE, img_np,
                int(RECT_CFG["cv_canny_low"]), int(RECT_CFG["cv_canny_high"]),
                float(RECT_CFG["cv_approx_epsilon"]), float(RECT_CFG["cv_area_min_ratio"]),
                float(RECT_CFG["cv_max_angle_cos"]), int(RECT_CFG["cv_gaussian_blur_size"]),
            )
        except Exception as e:
            print("cv_lite rectangle error:", e)
            rects = []
        for r in rects:
            try:
                c = self._candidate_quality(r)
            except Exception:
                c = None
            if c is not None:
                candidates.append(c)
        candidates.sort(key=lambda x: x["quality"], reverse=True)
        return candidates[:int(RECT_CFG["max_candidates"])]

    def _confirm_far_candidate(self, candidate):
        """远距离重捕获必须连续稳定，避免单帧误识别接管目标。"""
        cx, cy = candidate["center_det"]
        if self.pending_det is None:
            self.pending_det = (cx, cy)
            self.pending_count = 1
            return False
        d = math.sqrt((cx - self.pending_det[0]) ** 2 + (cy - self.pending_det[1]) ** 2)
        if d <= float(RECT_CFG["reacquire_stable_px"]):
            a = 0.55
            self.pending_det = (self.pending_det[0] + a * (cx - self.pending_det[0]), self.pending_det[1] + a * (cy - self.pending_det[1]))
            self.pending_count += 1
        else:
            self.pending_det = (cx, cy)
            self.pending_count = 1
        return self.pending_count >= int(RECT_CFG["reacquire_confirm_frames"])

    def step(self, img, img_np, dt):
        now = time.ticks_ms()
        candidates = self._detect_candidates(img_np)
        self.candidate_count = len(candidates)
        chosen = None
        if candidates:
            if self.center is not None:
                last_det = (logical_to_detect_x(self.center[0]), logical_to_detect_y(self.center[1]))
                gate = RECT_CFG["gate_track_px"] if self.locked else RECT_CFG["gate_recover_px"]
                best_cost = 1e9
                for c in candidates:
                    cx, cy = c["center_det"]
                    d = math.sqrt((cx - last_det[0]) ** 2 + (cy - last_det[1]) ** 2)
                    min_q = RECT_CFG["quality_low"] if self.locked else RECT_CFG["quality_recover"]
                    if d <= gate and c["quality"] >= min_q:
                        cost = d - 90.0 * c["quality"]
                        if cost < best_cost:
                            chosen = c
                            best_cost = cost
                if chosen is None and self.locked:
                    # 锁定后禁止远处最高分候选直接接管。只有连续丢失一段时间后，
                    # 才允许远候选通过“连续稳定确认”进入重捕获。
                    self.reject_far_count += 1
                    age = ticks_diff_ms(now, self.last_ms)
                    if age >= int(RECT_CFG["far_recover_delay_ms"]):
                        c0 = candidates[0]
                        if c0["quality"] >= RECT_CFG["quality_high"] and self._confirm_far_candidate(c0):
                            chosen = c0
                    # chosen仍为空时，本帧走短时保持，不更新center。
                elif chosen is not None:
                    self.pending_det = None
                    self.pending_count = 0
                    self.reject_far_count = 0
            else:
                # 初始捕获：要求高置信度，且连续帧由 acquire_frames 兜底。
                if candidates[0]["quality"] >= RECT_CFG["quality_high"]:
                    chosen = candidates[0]
        fresh = False
        if chosen is not None:
            cx_det, cy_det = chosen["center_det"]
            logical = logical_center_from_det(cx_det, cy_det)
            if self.center is not None:
                raw_vx = (logical[0] - self.center[0]) / max(0.004, dt)
                raw_vy = (logical[1] - self.center[1]) / max(0.004, dt)
                self.velocity = (0.70 * self.velocity[0] + 0.30 * raw_vx, 0.70 * self.velocity[1] + 0.30 * raw_vy)
            else:
                self.velocity = (0.0, 0.0)
            self.raw_center = logical
            self.center = logical
            self.box_det = chosen["box_det"]
            self.corners_det = chosen.get("corners_det")
            self.last_ms = now
            self.miss_frames = 0
            self.confidence = chosen["quality"]
            self.acquire_count += 1
            if self.acquire_count >= int(RECT_CFG["acquire_frames"]):
                self.locked = True
                self.state = 2
            else:
                self.state = 1
            fresh = True
        else:
            self.miss_frames += 1
            age = ticks_diff_ms(now, self.last_ms)
            if self.center is not None and age <= int(RECT_CFG["short_hold_ms"]):
                self.state = 3
                self.velocity = (0.86 * self.velocity[0], 0.86 * self.velocity[1])
            elif self.center is not None and age <= int(RECT_CFG["lost_ms"]):
                self.state = 3
                self.velocity = (0.65 * self.velocity[0], 0.65 * self.velocity[1])
            else:
                self.locked = False
                self.acquire_count = 0
                self.state = 0
                self.velocity = (0.0, 0.0)
                self.confidence = 0.0
                self.pending_det = None
                self.pending_count = 0
                if RECT_CFG.get("reset_after_lost", True):
                    self.center = None
                    self.raw_center = None
                    self.box_det = None
                    self.corners_det = None
        return {"center": self.center, "raw_center": self.raw_center, "velocity": self.velocity, "box_det": self.box_det, "corners_det": self.corners_det, "state": self.state, "fresh": fresh, "last_ms": self.last_ms, "age_ms": ticks_diff_ms(now, self.last_ms), "miss_frames": self.miss_frames, "candidate_count": self.candidate_count, "confidence": self.confidence, "reject_far_count": self.reject_far_count, "candidates": candidates}


class LaserSpotTrackerLite:
    def __init__(self):
        self.position_det = None
        self.position = None
        self.velocity_det = (0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.locked = False
        self.acquire_count = 0
        self.last_ms = time.ticks_ms()
        self.confidence = 0.0
        self.candidate_count = 0
        self.raw_blob_count = 0
        self.last_area = 0.0
        self.last_density = 0.0
        self.last_source = "NONE"
        self.ambiguous = 0
        self.last_candidates_det = []
        self.scan_count = 0

    def reset(self):
        self.position_det = None
        self.position = None
        self.velocity_det = (0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.locked = False
        self.acquire_count = 0
        self.confidence = 0.0
        self.candidate_count = 0
        self.raw_blob_count = 0
        self.last_source = "NONE"
        self.ambiguous = 0
        self.last_candidates_det = []
        self.scan_count = 0

    @staticmethod
    def _blob_value(blob, method, idx, default=0):
        try:
            return getattr(blob, method)()
        except Exception:
            try:
                return blob[idx]
            except Exception:
                return default

    @staticmethod
    def core_threshold():
        return (LASER_CFG["core_l_min"], LASER_CFG["core_l_max"], LASER_CFG["core_a_min"], LASER_CFG["core_a_max"], LASER_CFG["core_b_min"], LASER_CFG["core_b_max"])

    @staticmethod
    def halo_threshold():
        return (LASER_CFG["halo_l_min"], LASER_CFG["halo_l_max"], LASER_CFG["halo_a_min"], LASER_CFG["halo_a_max"], LASER_CFG["halo_b_min"], LASER_CFG["halo_b_max"])

    @staticmethod
    def _clip_roi(x0, y0, x1, y1):
        x0 = int(clamp(math.floor(x0), 0, LASER_DETECT_WIDTH - 1))
        y0 = int(clamp(math.floor(y0), 0, LASER_DETECT_HEIGHT - 1))
        x1 = int(clamp(math.ceil(x1), x0 + 1, LASER_DETECT_WIDTH))
        y1 = int(clamp(math.ceil(y1), y0 + 1, LASER_DETECT_HEIGHT))
        return (x0, y0, x1 - x0, y1 - y0)

    def _build_roi(self, rect_box_det):
        now = time.ticks_ms()
        if self.locked and self.position_det is not None:
            speed = math.sqrt(self.velocity_det[0] ** 2 + self.velocity_det[1] ** 2)
            g = clamp(float(LASER_CFG["gate_locked_px_det"]) + 0.05 * speed + 5.0, 20.0, float(LASER_CFG["dynamic_gate_max_det"]))
            px, py = self.position_det
            return self._clip_roi(px - g, py - g, px + g, py + g)
        if self.position_det is not None and ticks_diff_ms(now, self.last_ms) < int(LASER_CFG["anchor_keep_ms"]):
            g = float(LASER_CFG["anchor_gate_det"])
            px, py = self.position_det
            return self._clip_roi(px - g, py - g, px + g, py + g)
        if rect_box_det is not None:
            # 未锁定且尚无激光锚点时，大多数帧只搜索目标矩形附近；每隔若干
            # 次仍做一次全画面搜索，从而能够发现暂时落在矩形外的激光点。
            if self.position_det is None and self.scan_count % max(1, int(LASER_CFG["full_scan_period"])) == 0:
                return (0, 0, LASER_DETECT_WIDTH, LASER_DETECT_HEIGHT)
            x, y, w, h = rect_box_to_laser(rect_box_det)
            m = float(LASER_CFG["rect_roi_margin_det"])
            return self._clip_roi(x - m, y - m, x + w + m, y + h + m)
        return (0, 0, LASER_DETECT_WIDTH, LASER_DETECT_HEIGHT)

    def _find(self, img, threshold, roi, stride):
        try:
            # 单阈值无需做跨阈值合并；关闭merge可省去一次候选合并过程。
            stride = max(1, int(stride))
            return img.find_blobs([threshold], roi=roi, x_stride=stride, y_stride=stride, pixels_threshold=int(LASER_CFG["pixels_min"]), area_threshold=int(LASER_CFG["area_min"]), merge=False) or []
        except TypeError:
            return img.find_blobs([threshold], roi=roi, pixels_threshold=int(LASER_CFG["pixels_min"]), area_threshold=int(LASER_CFG["area_min"]), merge=False) or []

    def _candidate(self, blob, source, reference, gate):
        bx = float(self._blob_value(blob, "cx", 5, 0))
        by = float(self._blob_value(blob, "cy", 6, 0))
        bw = float(self._blob_value(blob, "w", 2, 0))
        bh = float(self._blob_value(blob, "h", 3, 0))
        pixels = float(self._blob_value(blob, "pixels", 4, bw * bh))
        area = max(1.0, bw * bh)
        if area < LASER_CFG["area_min"] or area > LASER_CFG["area_max"]:
            return None
        if bw <= 0 or bh <= 0 or bw > LASER_CFG["max_w_det"] or bh > LASER_CFG["max_h_det"]:
            return None
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if aspect > LASER_CFG["max_aspect"]:
            return None
        density = clamp(pixels / area, 0.0, 1.0)
        if density < LASER_CFG["min_density"]:
            return None
        d_score = 0.25
        if reference is not None:
            d = math.sqrt((bx - reference[0]) ** 2 + (by - reference[1]) ** 2)
            if d > gate:
                return None
            d_score = clamp(1.0 - d / max(1.0, gate), 0.0, 1.0)
        area_score = clamp(1.0 - abs(area - LASER_CFG["target_area"]) / 70.0, 0.0, 1.0)
        density_score = density
        source_bonus = 0.10 if source == "CORE" else 0.0
        score = clamp(0.42 * d_score + 0.28 * area_score + 0.20 * density_score + source_bonus, 0.0, 1.0)
        return (score, (bx, by), area, density, source)

    def detect(self, img, rect_box_det, dt):
        now = time.ticks_ms()
        self.scan_count = (self.scan_count + 1) & 0xFFFF
        roi = self._build_roi(rect_box_det)
        ref = self.position_det
        gate = float(LASER_CFG["gate_locked_px_det"] if self.locked else LASER_CFG["gate_reacquire_px_det"])
        blobs = []
        try:
            # 先以2像素步长寻找种子；find_blobs命中后仍会完整扩展连通域，
            # 因而候选中心、面积、密度和后续滤波都保持原逻辑。
            core = self._find(img, self.core_threshold(), roi, 2)
            # 极小的单像素光点可能落在步进网格之间。锁定状态下若快速扫描
            # 未命中，立即在同一小ROI内逐像素补扫，避免以准确率换帧率。
            if self.locked and not core:
                core = self._find(img, self.core_threshold(), roi, 1)
        except Exception:
            core = []
        blobs.extend((b, "CORE") for b in core)
        # 高频版只进行一次CORE扫描。旧版锁定后再扫HALO会把激光耗时翻倍。
        self.raw_blob_count = len(blobs)
        candidates = []
        for blob, src in blobs:
            c = self._candidate(blob, src, ref, gate)
            if c is not None:
                candidates.append(c)
        candidates.sort(key=lambda x: x[0], reverse=True)
        # 去重
        dedup = []
        for c in candidates:
            p = c[1]
            duplicate = False
            for k in dedup:
                if dist2(p, k[1]) <= 9.0:
                    duplicate = True
                    break
            if not duplicate:
                dedup.append(c)
        candidates = dedup
        self.candidate_count = len(candidates)
        self.last_candidates_det = [(float(c[1][0]), float(c[1][1]), float(c[2]), float(c[3])) for c in candidates[:8]]
        self.ambiguous = 0
        if not candidates:
            if ticks_diff_ms(now, self.last_ms) > int(LASER_CFG["anchor_keep_ms"]):
                self.locked = False
                self.acquire_count = 0
            self.last_source = "NONE"
            return self.position if self.locked else None
        if not self.locked:
            top = candidates[0][0]
            second = candidates[1][0] if len(candidates) > 1 else -1.0
            if len(candidates) > int(LASER_CFG["max_acquire_candidates"]) or (top - second) < float(LASER_CFG["ambiguity_margin"]):
                self.ambiguous = 1
                self.acquire_count = 0
                return None
            if top < float(LASER_CFG["min_acquire_score"]):
                return None
        elif candidates[0][0] < float(LASER_CFG["min_locked_score"]):
            return self.position
        score, measured_det, area, density, src = candidates[0]
        if self.position_det is not None:
            a = float(LASER_CFG["position_alpha"])
            filt = (self.position_det[0] + a * (measured_det[0] - self.position_det[0]), self.position_det[1] + a * (measured_det[1] - self.position_det[1]))
            raw_vx = (filt[0] - self.position_det[0]) / max(0.004, dt)
            raw_vy = (filt[1] - self.position_det[1]) / max(0.004, dt)
            va = float(LASER_CFG["velocity_alpha"])
            self.velocity_det = (self.velocity_det[0] + va * (raw_vx - self.velocity_det[0]), self.velocity_det[1] + va * (raw_vy - self.velocity_det[1]))
        else:
            filt = measured_det
            self.velocity_det = (0.0, 0.0)
        lim = float(LASER_CFG["velocity_limit_logical"])
        self.position_det = filt
        self.position = laser_center_to_logical(filt[0], filt[1])
        self.velocity = (clamp(self.velocity_det[0] * LASER_TO_LOGICAL_X, -lim, lim), clamp(self.velocity_det[1] * LASER_TO_LOGICAL_Y, -lim, lim))
        self.last_area = area
        self.last_density = density
        self.confidence = score
        self.last_source = src
        self.last_ms = now
        self.acquire_count += 1
        if self.acquire_count >= int(LASER_CFG["acquire_frames"]):
            self.locked = True
        return self.position if self.locked else None

class PlanBVisionNode:
    def __init__(self):
        self.uart = UARTLink()
        self.rect = RectTrackerLite()
        self.laser = LaserSpotTrackerLite()
        self.laser_output = LaserTTL35(
            pin_no=LASER_IO_PIN,
            active_level=LASER_ACTIVE_LEVEL,
        )

        now = time.ticks_ms()
        self.sequence = 0
        self.tx_frame = bytearray(AIM_FRAME_SIZE)
        self.frame_count = 0
        self.rect_frame_count = 0
        self.last_frame_ms = now
        self.last_plot_ms = time.ticks_add(now, -PLOT_INTERVAL_MS)
        self.fps = 0.0
        self.send_hz = 0.0
        self.last_send_ms = None

        self.rect_result = None
        self.laser_position = None
        self.rect_measured = False
        self.rect_fresh = False
        self.laser_measured = False
        self.laser_fresh = False
        self.state = 0
        self.valid_flags = 0

        self.capture_rect_ms = 0.0
        self.rect_detect_ms = 0.0
        self.capture_laser_ms = 0.0
        self.laser_detect_ms = 0.0
        self.uart_ms = 0.0
        self.packet_ms = 0.0
        self.display_ms = 0.0
        self.total_ms = 0.0

    def _update_fps(self, now):
        elapsed = ticks_diff_ms(now, self.last_frame_ms)
        self.last_frame_ms = now
        if elapsed <= 0:
            return 0.004
        instant = 1000.0 / float(elapsed)
        if self.fps <= 0.1:
            self.fps = instant
        else:
            self.fps = 0.85 * self.fps + 0.15 * instant
        return max(0.004, elapsed / 1000.0)

    def _classify_measurement(self, now, rect_result, laser_stamp_before):
        rect_position = rect_result.get("center") if rect_result else None
        rect_age = ticks_diff_ms(now, self.rect.last_ms) if rect_position is not None else 999999

        # LaserSpotTrackerLite只有真正接受了本帧候选才更新last_ms。
        self.laser_measured = (
            self.laser_position is not None
            and self.laser.last_ms != laser_stamp_before
            and ticks_diff_ms(now, self.laser.last_ms) <= LASER_FRESH_MAX_AGE_MS
        )
        self.laser_fresh = bool(
            self.laser_position is not None
            and ticks_diff_ms(now, self.laser.last_ms) <= LASER_FRESH_MAX_AGE_MS
        )
        self.rect_measured = bool(
            rect_result
            and rect_result.get("fresh")
            and rect_position is not None
            and rect_age <= RECT_FRESH_MAX_AGE_MS
        )
        # 隔帧矩形检测时，上一真实坐标在45 ms内仍作为当前可用观测。
        self.rect_fresh = bool(rect_position is not None and rect_age <= RECT_FRESH_MAX_AGE_MS)

        both_locked = bool(self.rect.locked and self.laser.locked)
        both_fresh = bool(self.rect_fresh and self.laser_fresh)

        flags = 0
        if self.rect_fresh:
            flags |= 0x01
        if self.laser_fresh:
            flags |= 0x02
        if both_locked:
            flags |= 0x04
        if both_fresh:
            flags |= 0x08
        # bit4~7在MSPM0中是保留位；任一置1都会导致整帧被拒收。
        self.valid_flags = flags

        laser_age = ticks_diff_ms(now, self.laser.last_ms)
        if both_fresh and both_locked:
            self.state = 2
        elif self.rect_fresh or self.laser_fresh:
            self.state = 1
        elif (
            rect_position is not None
            and self.laser.position is not None
            and rect_age <= HOLD_MAX_AGE_MS
            and laser_age <= HOLD_MAX_AGE_MS
        ):
            self.state = 3
        else:
            self.state = 0

    def _build_aim_frame(self, capture_ms):
        rect_position = self.rect_result.get("center") if self.rect_result else None
        laser_position = self.laser_position

        rect_x = coord_or_invalid(rect_position, 0, self.rect_fresh)
        rect_y = coord_or_invalid(rect_position, 1, self.rect_fresh)
        laser_x = coord_or_invalid(laser_position, 0, self.laser_fresh)
        laser_y = coord_or_invalid(laser_position, 1, self.laser_fresh)

        frame = self.tx_frame
        frame[0] = AIM_HEADER0
        frame[1] = AIM_HEADER1
        frame[2] = AIM_PROTOCOL_VERSION
        frame[3] = AIM_MSG_VISION_OBSERVATION
        write_le16(frame, 4, self.sequence)
        write_le32(frame, 6, capture_ms)
        write_le16(frame, 10, rect_x)
        write_le16(frame, 12, rect_y)
        write_le16(frame, 14, laser_x)
        write_le16(frame, 16, laser_y)
        frame[18] = self.valid_flags & 0x0F
        frame[19] = self.state & 0xFF
        crc = crc16_ccitt_false(frame, AIM_CRC_DATA_SIZE)
        write_le16(frame, 20, crc)
        return frame

    def send_observation(self, now, capture_ms):
        # 每个融合结果恰好发送一次；控制端需要的是最新观测，而不是定时抽样。
        self.last_plot_ms = now
        self.sequence = (self.sequence + 1) & 0xFFFF
        start_us = time.ticks_us()
        packet = self._build_aim_frame(capture_ms)
        self.packet_ms = perf_ms(start_us)
        start_us = time.ticks_us()
        self.uart.send_binary(packet)
        self.uart_ms = perf_ms(start_us)

        if self.last_send_ms is not None:
            elapsed = ticks_diff_ms(now, self.last_send_ms)
            if elapsed > 0:
                instant = 1000.0 / float(elapsed)
                self.send_hz = instant if self.send_hz <= 0.1 else 0.85 * self.send_hz + 0.15 * instant
        self.last_send_ms = now

    def draw_lcd(self, image):
        try:
            result = self.rect_result
            if result and result.get("box_det") is not None:
                x, y, w, h = result["box_det"]
                image.draw_rectangle(int(x), int(y), int(w), int(h), color=(0, 255, 0), thickness=2)

            rect_position = result.get("center") if result else None
            if rect_position is not None:
                rx = int(round(logical_to_detect_x(rect_position[0])))
                ry = int(round(logical_to_detect_y(rect_position[1])))
                image.draw_cross(rx, ry, color=(0, 255, 0), size=7, thickness=2)

            if self.laser_position is not None:
                lx = int(round(logical_to_detect_x(self.laser_position[0])))
                ly = int(round(logical_to_detect_y(self.laser_position[1])))
                image.draw_cross(lx, ly, color=(255, 0, 0), size=7, thickness=2)

            names = ("LOST", "ACQ", "TRACK", "HOLD")
            state_name = names[self.state] if 0 <= self.state < len(names) else "?"
            image.draw_string_advanced(
                2,
                2,
                12,
                "%s FPS%.1f TX%.1f F%02X" % (
                    state_name,
                    self.fps,
                    self.send_hz,
                    self.valid_flags,
                ),
                color=(255, 255, 255),
            )
            image.draw_string_advanced(
                2,
                17,
                11,
                "R%.1f L%.1f U%.1f T%.1f" % (
                    self.rect_detect_ms,
                    self.laser_detect_ms,
                    self.uart_ms,
                    self.total_ms,
                ),
                color=(255, 255, 0),
            )
            image.draw_string_advanced(
                2,
                31,
                11,
                "P%.1f CR%.1f CL%.1f" % (
                    self.packet_ms,
                    self.capture_rect_ms,
                    self.capture_laser_ms,
                ),
                color=(0, 255, 255),
            )
        except Exception:
            pass

    def run(self):
        sensor = None
        lcd_ok = False
        try:
            # 不允许失败后静默退回默认30/60 FPS模式；模式不支持时直接报错。
            sensor = Sensor(
                id=SENSOR_ID,
                width=SENSOR_INPUT_WIDTH,
                height=SENSOR_INPUT_HEIGHT,
                fps=SENSOR_FPS,
            )

            sensor.reset()
            sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT, chn=RECT_CHANNEL)
            sensor.set_pixformat(Sensor.RGB888, chn=RECT_CHANNEL)
            sensor.set_framesize(width=LASER_DETECT_WIDTH, height=LASER_DETECT_HEIGHT, chn=LASER_CHANNEL)
            sensor.set_pixformat(Sensor.RGB565, chn=LASER_CHANNEL)

            if ENABLE_LCD:
                try:
                    Display.init(
                        Display.ST7701,
                        width=640,
                        height=480,
                        to_ide=LCD_TO_IDE,
                        quality=LCD_QUALITY,
                    )
                    lcd_ok = True
                except Exception as error:
                    print("LCD init failed; continue headless:", error)

            MediaManager.init()
            sensor.run()
            time.sleep_ms(300)

            # 双通道预热，避免启动阶段的旧帧和曝光波动进入锁定器。
            for _ in range(12):
                os.exitpoint()
                try:
                    sensor.snapshot(chn=RECT_CHANNEL)
                except Exception:
                    pass
                try:
                    sensor.snapshot(chn=LASER_CHANNEL)
                except Exception:
                    pass

            if LASER_AUTO_ON:
                self.laser_output.on()

            display_x = int((640 - DETECT_WIDTH) / 2)
            display_y = int((480 - DETECT_HEIGHT) / 2)

            while True:
                os.exitpoint()
                loop_start_us = time.ticks_us()
                now = time.ticks_ms()
                capture_ms = now
                frame_dt = self._update_fps(now)

                rect_slot = (self.frame_count % RECT_SLOT_PERIOD == 0) or self.rect_result is None
                laser_stamp_before = self.laser.last_ms

                # 双输出通道必须持续出队。即使本时隙不运行对应算法，也要snapshot并
                # 及时释放该通道的帧，否则VB队列会积满并报snapshot failed(3)。
                start_us = time.ticks_us()
                rect_image = sensor.snapshot(chn=RECT_CHANNEL)
                self.capture_rect_ms = perf_ms(start_us)

                start_us = time.ticks_us()
                laser_image = sensor.snapshot(chn=LASER_CHANNEL)
                self.capture_laser_ms = perf_ms(start_us)

                if rect_slot:
                    # 矩形时隙不再同时跑激光；下一包复用最近35 ms内的激光坐标。
                    start_us = time.ticks_us()
                    rect_numpy = rect_image.to_numpy_ref()
                    self.rect_result = self.rect.step(rect_image, rect_numpy, frame_dt)
                    self.rect_detect_ms = perf_ms(start_us)
                    self.rect_frame_count += 1
                else:
                    # 只跳过矩形算法；通道帧在上方仍会持续出队。
                    self.rect_result["fresh"] = False
                    self.rect_result["age_ms"] = ticks_diff_ms(time.ticks_ms(), self.rect.last_ms)

                    start_us = time.ticks_us()
                    rect_box = self.rect_result.get("box_det") if self.rect_result else None
                    if self.laser_output.enabled:
                        self.laser_position = self.laser.detect(laser_image, rect_box, frame_dt)
                    else:
                        self.laser.reset()
                        self.laser_position = None
                    self.laser_detect_ms = perf_ms(start_us)

                now = time.ticks_ms()
                self._classify_measurement(now, self.rect_result, laser_stamp_before)
                self.send_observation(now, capture_ms)

                self.frame_count += 1
                self.display_ms = 0.0
                if (
                    lcd_ok
                    and rect_slot
                    and rect_image is not None
                    and self.rect_frame_count % max(1, LCD_REFRESH_EVERY_N_RECT_FRAMES) == 0
                ):
                    start_us = time.ticks_us()
                    self.draw_lcd(rect_image)
                    Display.show_image(rect_image, x=display_x, y=display_y)
                    self.display_ms = perf_ms(start_us)

                self.total_ms = perf_ms(loop_start_us)

                if self.frame_count % GC_CHECK_INTERVAL_FRAMES == 0:
                    try:
                        if gc.mem_free() < GC_FREE_THRESHOLD:
                            gc.collect()
                    except Exception:
                        gc.collect()

        except KeyboardInterrupt:
            pass
        except BaseException as error:
            print("Exception:", error)
            try:
                sys.print_exception(error)
            except Exception:
                pass
        finally:
            try:
                self.laser_output.off()
            except Exception:
                pass
            if isinstance(sensor, Sensor):
                try:
                    sensor.stop()
                except Exception:
                    pass
            if lcd_ok:
                try:
                    Display.deinit()
                except Exception:
                    pass
            try:
                os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
            except Exception:
                pass
            time.sleep_ms(100)
            try:
                MediaManager.deinit()
            except Exception:
                pass
            self.uart.close()
            self.laser_output.deinit()


if __name__ == "__main__":
    PlanBVisionNode().run()
