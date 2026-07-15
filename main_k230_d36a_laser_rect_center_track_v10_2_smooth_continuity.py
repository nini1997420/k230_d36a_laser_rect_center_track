# -*- coding: utf-8 -*-
"""
K230 + D36A 二维步进云台 矩形中心-激光点闭环 v10.2
固定周期连续平滑优化版 / Fixed-rate simple-control version

目标：解决 v5.7/v6.4 中 TRACK/COAST 高频切换、err 被置 0、target/output 跳变导致的卡顿。
核心策略：
1) 视觉测量与控制输出解耦；
2) 控制器固定 50 Hz 更新，步进 Ramp 固定 100 Hz 更新；
3) 短时漏检不把误差清零，不重置 PD，只做指令平滑衰减；
4) 严禁无效测量直接参与控制；
5) 使用连续 PD + 速度上限 + 起步补偿 + PWM 频率更新保护。

硬件固定：
    GPIO42 -> D36A STEP1（X）
    GPIO26 -> D36A DIR1 （X）
    GPIO43 -> D36A STEP2（Y）
    GPIO34 -> D36A DIR2 （Y）
    GPIO32 -> USB-TTL RX（K230 UART3 TX）
    GPIO33 <- USB-TTL TX（K230 UART3 RX）
    GPIO35 -> 激光 TTL/EN
    D36A EN1/EN2 -> D36A 自身 5V
    GND 共地

默认数据包：
[plot,err_x,err_y,x_target_hz,y_target_hz,x_output_hz,y_output_hz,rect_x,rect_y,state,fps]
state: 0=LOST/SEARCH, 1=ACQUIRE, 2=TRACK, 3=HOLD(short-lost), 4=STOP/FAULT
"""

import time
import os
import gc
import math
import sys

from machine import PWM, FPIOA, UART, Pin
from media.sensor import *
from media.display import *
from media.media import *

try:
    import cv_lite
except ImportError:
    raise RuntimeError("缺少 cv_lite，请使用当前 Yahboom/CanMV K230 固件。")

# ============================================================
# 1. 图像、通道、串口
# ============================================================
DETECT_WIDTH = 320
DETECT_HEIGHT = 240
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DETECT_TO_LOGICAL_X = CAMERA_WIDTH / float(DETECT_WIDTH)
DETECT_TO_LOGICAL_Y = CAMERA_HEIGHT / float(DETECT_HEIGHT)
IMAGE_SHAPE = [DETECT_HEIGHT, DETECT_WIDTH]

SENSOR_ID = 2
SENSOR_INPUT_WIDTH = 1280
SENSOR_INPUT_HEIGHT = 960
SENSOR_FPS = 90
RECT_CHANNEL = CAM_CHN_ID_0     # RGB888, cv_lite矩形
LASER_CHANNEL = CAM_CHN_ID_1    # RGB565, find_blobs激光

ENABLE_DISPLAY = False          # 追踪测试默认关闭显示，降低时序抖动
DISPLAY_TO_IDE = False
DISPLAY_QUALITY = 35
DISPLAY_EVERY_N_FRAMES = 5

ENABLE_UART = True
UART_BAUD = 115200
UART_TX_PIN = 32
UART_RX_PIN = 33
UART_ID = 3
PLOT_INTERVAL_MS = 30
STATUS_INTERVAL_MS = 1200
DIAG_INTERVAL_MS = 200
ENABLE_DIAG_PACKET = True
GC_CHECK_INTERVAL_FRAMES = 90
GC_FREE_THRESHOLD = 190000

# ============================================================
# 2. D36A STEP/DIR/激光 GPIO
# ============================================================
X_STEP_PIN = 42
X_DIR_PIN = 26
Y_STEP_PIN = 43
Y_DIR_PIN = 34

X_POSITIVE_DIR_LEVEL = 1
Y_POSITIVE_DIR_LEVEL = 0
X_REVERSE = False
Y_REVERSE = False

LASER_IO_PIN = 35
LASER_ACTIVE_LEVEL = 1
LASER_AUTO_ON_AFTER_CAMERA = True

USE_COMMON_ENABLE_PIN = False   # D36A EN1/EN2 已硬接驱动板5V
COMMON_ENABLE_PIN = 35
COMMON_ENABLE_ACTIVE_LEVEL = 0
ENABLE_SETTLE_MS = 20

STEP_DUTY_PERCENT = 50
PWM_INIT_FREQ_HZ = 100
DIR_SETUP_US = 8

MOTOR_FULL_STEPS_PER_REV = 200
MICROSTEP = 16
PULSES_PER_REV = MOTOR_FULL_STEPS_PER_REV * MICROSTEP
X_SOFT_LIMIT_STEPS = 6400.0
Y_SOFT_LIMIT_STEPS = 2844.0

# PWM.freq() 保护：避免高频微小变化反复重装载硬件计数器
PWM_FREQ_REL_DELTA = 0.018
PWM_FREQ_ABS_DELTA_HZ = 3
PWM_FREQ_MAX_HOLD_MS = 70

# ============================================================
# 3. 矩形、激光检测参数
# ============================================================
RECT_CFG = {
    "min_area_detect": 120.0,
    "max_area_ratio": 0.46,
    "min_w_detect": 12,
    "min_h_detect": 9,
    "aspect_min": 1.05,
    "aspect_max": 2.38,
    "target_aspect": 1.50,
    "quality_high": 0.36,
    "quality_low": 0.10,
    "quality_recover": 0.22,
    "max_candidates": 12,

    # v10.2：强目标身份锁定。v10数据中的乱抖主要来自矩形候选跳到远处干扰框，
    # 因此锁定后不允许直接接受远离上一帧轨迹的全局最高分候选。
    "gate_track_px": 78.0,
    "gate_recover_px": 175.0,
    "far_recover_delay_ms": 300,
    "reacquire_stable_px": 36.0,
    "reacquire_confirm_frames": 2,
    "reset_after_lost": True,
    "cv_canny_low": 22,
    "cv_canny_high": 84,
    "cv_approx_epsilon": 0.023,
    "cv_area_min_ratio": 0.0010,
    "cv_max_angle_cos": 0.46,
    "cv_gaussian_blur_size": 3,
    "jump_guard_px": 120.0,
    "acquire_frames": 2,
    "short_hold_ms": 240,
    "lost_ms": 650,
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
    "gate_locked_px_det": 26.0,
    "gate_reacquire_px_det": 130.0,
    "acquire_jump_px_det": 34.0,
    "acquire_frames": 2,
    "max_acquire_candidates": 8,
    "min_acquire_score": 0.42,
    "min_locked_score": 0.24,
    "ambiguity_margin": 0.06,
    "position_alpha": 0.82,
    "velocity_alpha": 0.30,
    "velocity_limit_logical": 1000.0,
    "anchor_keep_ms": 1900,
    "anchor_gate_det": 35.0,
    "dynamic_gate_max_det": 50.0,
    "rect_roi_margin_det": 50.0,
    "full_scan_fallback_ms": 800,
}

# ============================================================
# 4. v10 固定周期控制参数
# ============================================================
CONTROL_PERIOD_MS = 20          # 50Hz 控制器
RAMP_PERIOD_MS = 10             # 100Hz 步进Ramp
MEASURE_FRESH_MS = 90           # 两个测量都在此时间内有效才全速TRACK
MEASURE_HOLD_MS = 260           # 短时漏检：保留误差但衰减目标频率
MEASURE_LOST_MS = 560           # 超过该时间：目标频率归零，Ramp减速停

CTRL_CFG = {
    # v6.4的最大频率较激进；v10先以连续性为第一目标，留足硬件余量。
    "x_min_hz": 12.0,
    "y_min_hz": 10.0,
    "x_max_hz": 1300.0,
    "y_max_hz": 1050.0,
    "x_accel_hz_s": 4300.0,
    "y_accel_hz_s": 3900.0,
    "x_decel_hz_s": 6100.0,
    "y_decel_hz_s": 5500.0,

    # PD，输出单位 Hz；不使用积分，避免漏检/重捕获后积分残留。
    "x_kp": 5.25,
    "y_kp": 4.35,
    "x_kd": 0.080,
    "y_kd": 0.080,
    "d_filter_tau_s": 0.085,

    # 误差滤波；大误差快，小误差稳。
    "err_alpha_near": 0.26,
    "err_alpha_mid": 0.48,
    "err_alpha_far": 0.72,
    "near_error_px": 16.0,
    "mid_error_px": 55.0,
    "far_error_px": 140.0,

    # 中心停止迟滞，防止反复启停。
    "x_stop_enter_px": 5.0,
    "x_stop_exit_px": 9.0,
    "y_stop_enter_px": 5.0,
    "y_stop_exit_px": 9.0,

    # 连续速度上限。
    "x_near_cap_hz": 68.0,
    "y_near_cap_hz": 62.0,
    "x_mid_cap_hz": 470.0,
    "y_mid_cap_hz": 380.0,
    "x_far_cap_hz": 980.0,
    "y_far_cap_hz": 780.0,

    # 有效误差足够大才补最低步频；中心附近不强行启动。
    "x_start_error_px": 10.0,
    "y_start_error_px": 10.0,

    # 测量丢失时目标指令衰减，不硬切0。
    "hold_decay_1": 0.82,
    "hold_decay_2": 0.55,
    "hold_cap_x_hz": 220.0,
    "hold_cap_y_hz": 180.0,

    # v10.2：测量异常跳变保护。第一帧大跳变先当作可疑测量，只衰减输出；
    # 连续确认后才接受，避免单帧误识别导致云台乱甩。
    "raw_jump_reject_px": 115.0,
    "raw_jump_confirm_frames": 2,
    "outlier_decay": 0.45,
    "outlier_cap_x_hz": 160.0,
    "outlier_cap_y_hz": 135.0,

    # v10.2：目标频率二级限斜率。StepperAxis已有Ramp，这里再限制控制目标的突然变化，
    # 让 target_hz 本身连续，减少重捕获/短时漏检后的冲击。
    "x_target_slew_hz_s": 3600.0,
    "y_target_slew_hz_s": 3200.0,
    "x_zero_cross_stop_px": 18.0,
    "y_zero_cross_stop_px": 18.0,

    # 安全边界：激光靠近画面边缘时禁止继续向外走。
    "laser_edge_x_px": 22.0,
    "laser_edge_y_px": 18.0,
}

# ============================================================
# 5. 工具函数
# ============================================================
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
            print("External UART ready: GPIO%d(TX) GPIO%d(RX) UART3 @ %d" % (UART_TX_PIN, UART_RX_PIN, UART_BAUD))
        except Exception as e:
            print("UART3 init failed; continue without external UART:", e)
            self.uart = None

    def is_ready(self):
        return self.uart is not None

    def send(self, text):
        print(text)
        if self.uart is None:
            return False
        try:
            self.uart.write((str(text) + "\r\n").encode("utf-8"))
            self.tx_packets += 1
            return True
        except Exception as e:
            print("UART write failed:", e)
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


class CommonEnable:
    def __init__(self):
        self.pin = None
        self.enabled = True
        if USE_COMMON_ENABLE_PIN:
            inactive = 0 if COMMON_ENABLE_ACTIVE_LEVEL else 1
            self.pin = make_gpio_output(COMMON_ENABLE_PIN, inactive)
            self.disable()

    def enable(self):
        self.enabled = True
        if self.pin is not None:
            write_pin(self.pin, COMMON_ENABLE_ACTIVE_LEVEL)
            try:
                time.sleep_ms(ENABLE_SETTLE_MS)
            except Exception:
                pass

    def disable(self):
        self.enabled = False
        if self.pin is not None:
            write_pin(self.pin, 0 if COMMON_ENABLE_ACTIVE_LEVEL else 1)

    def deinit(self):
        self.disable()


class LaserTTL35:
    def __init__(self, pin_no=LASER_IO_PIN, active_level=LASER_ACTIVE_LEVEL):
        self.pin_no = int(pin_no)
        self.active_level = 1 if active_level else 0
        self.pin = make_gpio_output(self.pin_no, 1 - self.active_level)
        self.enabled = False
        self.off()
        print("Laser TTL GPIO%d ready active=%d default=OFF" % (self.pin_no, self.active_level))

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


class StepperAxis:
    def __init__(self, name, step_pin, dir_pin, positive_dir_level, reverse, min_hz, max_hz, accel_hz_s, decel_hz_s, soft_limit_steps):
        self.name = str(name)
        self.step_pin = int(step_pin)
        self.dir_pin_number = int(dir_pin)
        self.positive_dir_level = 1 if positive_dir_level else 0
        self.reverse = bool(reverse)
        self.min_hz = float(min_hz)
        self.max_hz = float(max_hz)
        self.accel_hz_s = float(accel_hz_s)
        self.decel_hz_s = float(decel_hz_s)
        self.soft_limit_steps = max(0.0, float(soft_limit_steps))
        self.target_hz = 0.0
        self.output_hz = 0.0
        self.applied_hz = 0.0
        self.virtual_steps = 0.0
        self.direction_sign = 0
        self.running = False
        self.last_pwm_freq = 0
        self.last_freq_update_ms = time.ticks_ms()
        self.limit_hit = 0
        self.enabled = True
        self.dir_pin = make_gpio_output(self.dir_pin_number, self.positive_dir_level)
        try:
            self.pwm = PWM(self.step_pin, freq=int(PWM_INIT_FREQ_HZ), duty=0)
        except TypeError:
            self.pwm = PWM(self.step_pin, int(PWM_INIT_FREQ_HZ), 0)
        self._set_direction(1)
        self._stop_pwm()
        limit_deg = self.soft_limit_steps * 360.0 / float(PULSES_PER_REV) if self.soft_limit_steps > 0.0 else 0.0
        print("[%s] STEP GPIO%d / DIR GPIO%d / min=%.0f max=%.0fHz limit=%.0f steps (±%.1f deg)" % (
            self.name, self.step_pin, self.dir_pin_number, self.min_hz, self.max_hz, self.soft_limit_steps, limit_deg))

    def _physical_sign(self, logical_sign):
        return -logical_sign if self.reverse else logical_sign

    def _set_direction(self, logical_sign):
        logical_sign = 1 if logical_sign >= 0 else -1
        physical = self._physical_sign(logical_sign)
        level = self.positive_dir_level if physical > 0 else (1 - self.positive_dir_level)
        if logical_sign != self.direction_sign:
            self._stop_pwm()
            write_pin(self.dir_pin, level)
            try:
                time.sleep_us(DIR_SETUP_US)
            except Exception:
                pass
            self.direction_sign = logical_sign

    def _stop_pwm(self):
        try:
            self.pwm.duty(0)
        except Exception:
            pass
        self.running = False
        self.applied_hz = 0.0

    def _should_update_freq(self, freq):
        if freq != self.last_pwm_freq and self.last_pwm_freq <= 0:
            return True
        diff = abs(int(freq) - int(self.last_pwm_freq))
        if diff >= PWM_FREQ_ABS_DELTA_HZ and diff / max(1.0, float(self.last_pwm_freq)) >= PWM_FREQ_REL_DELTA:
            return True
        if ticks_diff_ms(time.ticks_ms(), self.last_freq_update_ms) >= PWM_FREQ_MAX_HOLD_MS and diff > 0:
            return True
        return False

    def _apply_pwm(self, signed_hz):
        s = sign_of(signed_hz)
        mag = abs(float(signed_hz))
        if s == 0 or mag < self.min_hz or not self.enabled:
            self._stop_pwm()
            return 0.0
        mag = clamp(mag, self.min_hz, self.max_hz)
        self._set_direction(s)
        freq = max(1, int(round(mag)))
        if self._should_update_freq(freq):
            self.pwm.freq(freq)
            self.last_pwm_freq = freq
            self.last_freq_update_ms = time.ticks_ms()
        if not self.running:
            self.pwm.duty(int(STEP_DUTY_PERCENT))
            self.running = True
        # 如果频率被保护未更新，则实际频率仍按 last_pwm_freq 估算。
        actual_freq = self.last_pwm_freq if self.last_pwm_freq > 0 else freq
        self.applied_hz = float(s * actual_freq)
        return self.applied_hz

    def _soft_limit_allows(self, signed_hz):
        self.limit_hit = 0
        if self.soft_limit_steps <= 0.0:
            return signed_hz
        if signed_hz > 0.0 and self.virtual_steps >= self.soft_limit_steps:
            self.limit_hit = 1
            return 0.0
        if signed_hz < 0.0 and self.virtual_steps <= -self.soft_limit_steps:
            self.limit_hit = -1
            return 0.0
        return signed_hz

    def set_target_hz(self, signed_hz):
        if not self.enabled:
            self.target_hz = 0.0
            return 0.0
        value = clamp(float(signed_hz), -self.max_hz, self.max_hz)
        value = self._soft_limit_allows(value)
        if 0.0 < abs(value) < self.min_hz:
            value = self.min_hz if value > 0.0 else -self.min_hz
        self.target_hz = value
        return self.target_hz

    def update(self, dt):
        dt = clamp(float(dt), 0.001, 0.250)
        old = self.applied_hz
        target = self._soft_limit_allows(self.target_hz)
        self.target_hz = target
        current = self.output_hz
        cs = sign_of(current)
        ts = sign_of(target)
        if cs != 0 and ts != 0 and cs != ts:
            next_value = move_toward(current, 0.0, self.decel_hz_s * dt)
        elif ts == 0:
            next_value = move_toward(current, 0.0, self.decel_hz_s * dt)
        else:
            rate = self.accel_hz_s if abs(target) > abs(current) else self.decel_hz_s
            next_value = move_toward(current, target, rate * dt)
        if abs(next_value) < 0.5:
            next_value = 0.0
        self.output_hz = self._soft_limit_allows(clamp(next_value, -self.max_hz, self.max_hz))
        new = self._apply_pwm(self.output_hz)
        self.virtual_steps += 0.5 * (old + new) * dt
        if self.soft_limit_steps > 0.0:
            self.virtual_steps = clamp(self.virtual_steps, -self.soft_limit_steps - 1.0, self.soft_limit_steps + 1.0)
        return self.applied_hz

    def hard_stop(self):
        self.target_hz = 0.0
        self.output_hz = 0.0
        self._stop_pwm()

    def zero_virtual_position(self):
        self.virtual_steps = 0.0
        self.limit_hit = 0

    def deinit(self):
        self.hard_stop()
        try:
            self.pwm.deinit()
        except Exception:
            pass

# ============================================================
# 7. 检测器：矩形 / 激光
# ============================================================
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
        x0 = int(clamp(math.floor(x0), 0, DETECT_WIDTH - 1))
        y0 = int(clamp(math.floor(y0), 0, DETECT_HEIGHT - 1))
        x1 = int(clamp(math.ceil(x1), x0 + 1, DETECT_WIDTH))
        y1 = int(clamp(math.ceil(y1), y0 + 1, DETECT_HEIGHT))
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
        if rect_box_det is not None and ticks_diff_ms(now, self.last_ms) < int(LASER_CFG["full_scan_fallback_ms"]):
            x, y, w, h = rect_box_det
            m = float(LASER_CFG["rect_roi_margin_det"])
            return self._clip_roi(x - m, y - m, x + w + m, y + h + m)
        return (0, 0, DETECT_WIDTH, DETECT_HEIGHT)

    def _find(self, img, threshold, roi):
        try:
            return img.find_blobs([threshold], roi=roi, x_stride=1, y_stride=1, pixels_threshold=int(LASER_CFG["pixels_min"]), area_threshold=int(LASER_CFG["area_min"]), merge=True, margin=int(LASER_CFG["merge_margin"])) or []
        except TypeError:
            return img.find_blobs([threshold], roi=roi, pixels_threshold=int(LASER_CFG["pixels_min"]), area_threshold=int(LASER_CFG["area_min"]), merge=True, margin=int(LASER_CFG["merge_margin"])) or []

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
        roi = self._build_roi(rect_box_det)
        ref = self.position_det
        gate = float(LASER_CFG["gate_locked_px_det"] if self.locked else LASER_CFG["gate_reacquire_px_det"])
        blobs = []
        try:
            core = self._find(img, self.core_threshold(), roi)
        except Exception as e:
            print("laser core find error:", e)
            core = []
        blobs.extend((b, "CORE") for b in core)
        # 已锁定时允许宽阈值在小ROI内补偿红色光晕，不在全屏宽搜。
        if self.locked and self.position_det is not None:
            try:
                halo = self._find(img, self.halo_threshold(), roi)
            except Exception:
                halo = []
            blobs.extend((b, "HALO") for b in halo)
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
        self.position = logical_center_from_det(filt[0], filt[1])
        self.velocity = (clamp(self.velocity_det[0] * DETECT_TO_LOGICAL_X, -lim, lim), clamp(self.velocity_det[1] * DETECT_TO_LOGICAL_Y, -lim, lim))
        self.last_area = area
        self.last_density = density
        self.confidence = score
        self.last_source = src
        self.last_ms = now
        self.acquire_count += 1
        if self.acquire_count >= int(LASER_CFG["acquire_frames"]):
            self.locked = True
        return self.position if self.locked else None

# ============================================================
# 8. v10 固定周期控制器
# ============================================================
class FixedRatePDController:
    def __init__(self):
        self.err_x = None
        self.err_y = None
        self.prev_err_x = None
        self.prev_err_y = None
        self.der_x = 0.0
        self.der_y = 0.0
        self.x_hold = False
        self.y_hold = False
        self.last_target_x = 0.0
        self.last_target_y = 0.0
        self.slew_target_x = 0.0
        self.slew_target_y = 0.0
        self.state_code = 0
        self.valid_ms = 999999
        self.last_valid_ms = 0
        self.outlier_count = 0

    def reset(self):
        self.err_x = None
        self.err_y = None
        self.prev_err_x = None
        self.prev_err_y = None
        self.der_x = 0.0
        self.der_y = 0.0
        self.x_hold = False
        self.y_hold = False
        self.last_target_x = 0.0
        self.last_target_y = 0.0
        self.slew_target_x = 0.0
        self.slew_target_y = 0.0
        self.state_code = 0
        self.outlier_count = 0

    @staticmethod
    def _err_alpha(abs_e):
        if abs_e <= CTRL_CFG["near_error_px"]:
            return CTRL_CFG["err_alpha_near"]
        if abs_e <= CTRL_CFG["mid_error_px"]:
            return CTRL_CFG["err_alpha_mid"]
        return CTRL_CFG["err_alpha_far"]

    @staticmethod
    def _cap(abs_e, near_cap, mid_cap, far_cap):
        near_e = CTRL_CFG["near_error_px"]
        mid_e = CTRL_CFG["mid_error_px"]
        far_e = CTRL_CFG["far_error_px"]
        if abs_e <= near_e:
            return near_cap
        if abs_e <= mid_e:
            r = (abs_e - near_e) / max(1.0, mid_e - near_e)
            return near_cap + r * (mid_cap - near_cap)
        if abs_e <= far_e:
            r = (abs_e - mid_e) / max(1.0, far_e - mid_e)
            return mid_cap + r * (far_cap - mid_cap)
        return far_cap

    def _filter_error(self, raw_x, raw_y):
        if self.err_x is None:
            self.err_x = float(raw_x)
            self.err_y = float(raw_y)
            return
        ax = self._err_alpha(abs(raw_x))
        ay = self._err_alpha(abs(raw_y))
        # 跨零或目标突然远离时快速跟随，避免滤波延迟造成反向输出。
        if raw_x * self.err_x <= 0.0 or abs(raw_x) > abs(self.err_x) + 8.0:
            ax = max(ax, 0.82)
        if raw_y * self.err_y <= 0.0 or abs(raw_y) > abs(self.err_y) + 8.0:
            ay = max(ay, 0.82)
        self.err_x += ax * (float(raw_x) - self.err_x)
        self.err_y += ay * (float(raw_y) - self.err_y)

    def _axis_pd(self, axis_name, e, dt):
        if axis_name == "x":
            enter = CTRL_CFG["x_stop_enter_px"]
            exit_e = CTRL_CFG["x_stop_exit_px"]
            kp = CTRL_CFG["x_kp"]
            kd = CTRL_CFG["x_kd"]
            near_cap = CTRL_CFG["x_near_cap_hz"]
            mid_cap = CTRL_CFG["x_mid_cap_hz"]
            far_cap = CTRL_CFG["x_far_cap_hz"]
            min_hz = CTRL_CFG["x_min_hz"]
            start_e = CTRL_CFG["x_start_error_px"]
            prev = self.prev_err_x
            holding = self.x_hold
        else:
            enter = CTRL_CFG["y_stop_enter_px"]
            exit_e = CTRL_CFG["y_stop_exit_px"]
            kp = CTRL_CFG["y_kp"]
            kd = CTRL_CFG["y_kd"]
            near_cap = CTRL_CFG["y_near_cap_hz"]
            mid_cap = CTRL_CFG["y_mid_cap_hz"]
            far_cap = CTRL_CFG["y_far_cap_hz"]
            min_hz = CTRL_CFG["y_min_hz"]
            start_e = CTRL_CFG["y_start_error_px"]
            prev = self.prev_err_y
            holding = self.y_hold
        ae = abs(float(e))
        if holding:
            if ae <= exit_e:
                target = 0.0
                if axis_name == "x": self.x_hold = True
                else: self.y_hold = True
                return target
            if axis_name == "x": self.x_hold = False
            else: self.y_hold = False
        elif ae <= enter:
            if axis_name == "x": self.x_hold = True
            else: self.y_hold = True
            return 0.0

        raw_d = 0.0 if prev is None else (float(e) - float(prev)) / max(0.004, dt)
        a = dt / (CTRL_CFG["d_filter_tau_s"] + dt)
        if axis_name == "x":
            self.der_x += a * (raw_d - self.der_x)
            d = self.der_x
        else:
            self.der_y += a * (raw_d - self.der_y)
            d = self.der_y
        target = kp * float(e) + kd * d
        cap = self._cap(ae, near_cap, mid_cap, far_cap)
        target = clamp(target, -cap, cap)
        if ae >= start_e and 0.0 < abs(target) < min_hz:
            target = min_hz if target > 0.0 else -min_hz
        # 小误差跨零后先停，不强行反向。
        zero_cross_stop = CTRL_CFG["x_zero_cross_stop_px"] if axis_name == "x" else CTRL_CFG["y_zero_cross_stop_px"]
        if prev is not None and prev * e < 0.0 and ae < zero_cross_stop:
            target = 0.0
        return target

    def _shape_target(self, tx, ty, dt):
        max_dx = float(CTRL_CFG["x_target_slew_hz_s"]) * max(0.004, float(dt))
        max_dy = float(CTRL_CFG["y_target_slew_hz_s"]) * max(0.004, float(dt))
        self.slew_target_x = move_toward(self.slew_target_x, float(tx), max_dx)
        self.slew_target_y = move_toward(self.slew_target_y, float(ty), max_dy)
        return self.slew_target_x, self.slew_target_y

    def update(self, rect_pos, rect_age_ms, laser_pos, laser_age_ms, dt, laser_edge_pos=None):
        now = time.ticks_ms()
        valid = rect_pos is not None and laser_pos is not None and rect_age_ms <= MEASURE_FRESH_MS and laser_age_ms <= MEASURE_FRESH_MS
        short_hold = rect_pos is not None and laser_pos is not None and rect_age_ms <= MEASURE_HOLD_MS and laser_age_ms <= MEASURE_HOLD_MS
        if valid:
            raw_x = float(rect_pos[0] - laser_pos[0])
            raw_y = float(rect_pos[1] - laser_pos[1])
            if self.err_x is not None:
                jump = math.sqrt((raw_x - self.err_x) ** 2 + (raw_y - self.err_y) ** 2)
                if jump > float(CTRL_CFG["raw_jump_reject_px"]):
                    self.outlier_count += 1
                    if self.outlier_count < int(CTRL_CFG["raw_jump_confirm_frames"]):
                        tx = clamp(self.last_target_x * CTRL_CFG["outlier_decay"], -CTRL_CFG["outlier_cap_x_hz"], CTRL_CFG["outlier_cap_x_hz"])
                        ty = clamp(self.last_target_y * CTRL_CFG["outlier_decay"], -CTRL_CFG["outlier_cap_y_hz"], CTRL_CFG["outlier_cap_y_hz"])
                        tx, ty = self._shape_target(tx, ty, dt)
                        self.last_target_x = tx
                        self.last_target_y = ty
                        self.state_code = 3
                        return tx, ty, self.state_code
                else:
                    self.outlier_count = 0
            self._filter_error(raw_x, raw_y)
            tx = self._axis_pd("x", self.err_x, dt)
            ty = self._axis_pd("y", self.err_y, dt)
            self.prev_err_x = self.err_x
            self.prev_err_y = self.err_y
            tx, ty = self._shape_target(tx, ty, dt)
            self.last_target_x = tx
            self.last_target_y = ty
            self.last_valid_ms = now
            self.state_code = 2
        elif short_hold and self.err_x is not None:
            age = max(float(rect_age_ms), float(laser_age_ms))
            decay = CTRL_CFG["hold_decay_1"] if age <= MEASURE_HOLD_MS * 0.55 else CTRL_CFG["hold_decay_2"]
            tx = clamp(self.last_target_x * decay, -CTRL_CFG["hold_cap_x_hz"], CTRL_CFG["hold_cap_x_hz"])
            ty = clamp(self.last_target_y * decay, -CTRL_CFG["hold_cap_y_hz"], CTRL_CFG["hold_cap_y_hz"])
            tx, ty = self._shape_target(tx, ty, dt)
            self.last_target_x = tx
            self.last_target_y = ty
            self.state_code = 3
        else:
            tx = 0.0
            ty = 0.0
            tx, ty = self._shape_target(0.0, 0.0, dt)
            if abs(tx) < 1.0:
                tx = 0.0
                self.slew_target_x = 0.0
            if abs(ty) < 1.0:
                ty = 0.0
                self.slew_target_y = 0.0
            self.last_target_x = tx
            self.last_target_y = ty
            self.state_code = 0
            if self.err_x is not None and ticks_diff_ms(now, self.last_valid_ms) > MEASURE_LOST_MS:
                self.err_x = None
                self.err_y = None
                self.prev_err_x = None
                self.prev_err_y = None
                self.der_x = 0.0
                self.der_y = 0.0
                self.x_hold = False
                self.y_hold = False

        if laser_edge_pos is not None:
            lx, ly = laser_edge_pos
            if lx <= CTRL_CFG["laser_edge_x_px"] and tx < 0.0:
                tx = 0.0
            if lx >= CAMERA_WIDTH - CTRL_CFG["laser_edge_x_px"] and tx > 0.0:
                tx = 0.0
            if ly <= CTRL_CFG["laser_edge_y_px"] and ty < 0.0:
                ty = 0.0
            if ly >= CAMERA_HEIGHT - CTRL_CFG["laser_edge_y_px"] and ty > 0.0:
                ty = 0.0
        return tx, ty, self.state_code

# ============================================================
# 9. 主程序
# ============================================================
class K230D36ALaserRectTrackerV10:
    def __init__(self):
        self.common_enable = CommonEnable()
        self.x_axis = StepperAxis("X", X_STEP_PIN, X_DIR_PIN, X_POSITIVE_DIR_LEVEL, X_REVERSE, CTRL_CFG["x_min_hz"], CTRL_CFG["x_max_hz"], CTRL_CFG["x_accel_hz_s"], CTRL_CFG["x_decel_hz_s"], X_SOFT_LIMIT_STEPS)
        self.y_axis = StepperAxis("Y", Y_STEP_PIN, Y_DIR_PIN, Y_POSITIVE_DIR_LEVEL, Y_REVERSE, CTRL_CFG["y_min_hz"], CTRL_CFG["y_max_hz"], CTRL_CFG["y_accel_hz_s"], CTRL_CFG["y_decel_hz_s"], Y_SOFT_LIMIT_STEPS)
        self.common_enable.enable()
        self.laser_output = LaserTTL35()
        self.uart = UARTLink()
        self.rect = RectTrackerLite()
        self.laser = LaserSpotTrackerLite()
        self.controller = FixedRatePDController()
        self.tracking_enabled = True
        self.estop = False
        self.laser_x_reverse = False
        self.laser_y_reverse = False
        now = time.ticks_ms()
        self.last_plot_ms = now
        self.last_status_ms = now
        self.last_diag_ms = now
        self.last_control_ms = now
        self.last_ramp_ms = now
        self.last_frame_ms = now
        self.fps = 0.0
        self.frame_count = 0
        self.last_result = None
        self.current_laser_pos = None
        self.last_x_target = 0.0
        self.last_y_target = 0.0
        self.perf_capture_ms = 0.0
        self.perf_rect_ms = 0.0
        self.perf_laser_ms = 0.0
        self.perf_total_ms = 0.0

    def stop_motion(self, hard=False):
        self.last_x_target = 0.0
        self.last_y_target = 0.0
        if hard:
            self.x_axis.hard_stop()
            self.y_axis.hard_stop()
        else:
            self.x_axis.set_target_hz(0.0)
            self.y_axis.set_target_hz(0.0)

    def emergency_stop(self):
        self.estop = True
        self.stop_motion(hard=True)
        self.common_enable.disable()

    def restart(self):
        self.common_enable.enable()
        self.estop = False
        self.controller.reset()
        self.rect.reset()
        self.laser.reset()
        self.stop_motion(hard=True)

    def _jog_axis_blocking(self, axis, hz, run_ms):
        axis.set_target_hz(float(hz))
        start = time.ticks_ms()
        last = start
        while ticks_diff_ms(time.ticks_ms(), start) < int(run_ms):
            now = time.ticks_ms()
            dt = max(0.001, ticks_diff_ms(now, last) / 1000.0)
            last = now
            axis.update(dt)
            time.sleep_ms(10)
        axis.set_target_hz(0.0)
        for _ in range(30):
            axis.update(0.01)
            time.sleep_ms(10)

    def motor_test(self):
        seq = (("X+", self.x_axis, 120.0), ("X-", self.x_axis, -120.0), ("Y+", self.y_axis, 120.0), ("Y-", self.y_axis, -120.0))
        self.uart.send("[motor,test,start]")
        for label, axis, hz in seq:
            self.uart.send("[motor,test,%s,%.0f,600]" % (label, hz))
            self._jog_axis_blocking(axis, hz, 600)
            time.sleep_ms(200)
        self.stop_motion(hard=True)
        self.x_axis.zero_virtual_position()
        self.y_axis.zero_virtual_position()
        self.uart.send("[motor,test,end]")

    def handle_uart(self):
        global X_REVERSE, Y_REVERSE
        for p in self.uart.read_packets():
            cmd = p[0].lower()
            try:
                if cmd in ("key", "sv") and len(p) >= 2:
                    act = p[1].lower()
                    if act in ("stop", "estop"):
                        self.emergency_stop()
                        self.uart.send("[ack,stop]")
                    elif act in ("start", "restart"):
                        self.restart()
                        self.tracking_enabled = True
                        self.uart.send("[ack,start]")
                elif cmd == "motor" and len(p) >= 2:
                    act = p[1].lower()
                    if act == "test":
                        self.motor_test()
                    elif act == "jog" and len(p) >= 5:
                        axis = self.x_axis if p[2].lower().startswith("x") else self.y_axis
                        hz = float(p[3])
                        run_ms = int(float(p[4]))
                        self._jog_axis_blocking(axis, hz, run_ms)
                elif cmd == "slider" and len(p) >= 3:
                    name = p[1].strip().lower()
                    value = float(p[2])
                    self.apply_slider(name, value)
            except Exception as e:
                self.uart.send("[ack,error,%s]" % str(e))

    def apply_slider(self, name, value):
        if name in ("xkp", "x_kp"):
            CTRL_CFG["x_kp"] = clamp(value, 0.0, 20.0)
        elif name in ("ykp", "y_kp"):
            CTRL_CFG["y_kp"] = clamp(value, 0.0, 20.0)
        elif name in ("xkd", "x_kd"):
            CTRL_CFG["x_kd"] = clamp(value, 0.0, 5.0)
        elif name in ("ykd", "y_kd"):
            CTRL_CFG["y_kd"] = clamp(value, 0.0, 5.0)
        elif name in ("xmaxhz", "x_max_hz"):
            self.x_axis.max_hz = clamp(value, self.x_axis.min_hz, 3000.0)
        elif name in ("ymaxhz", "y_max_hz"):
            self.y_axis.max_hz = clamp(value, self.y_axis.min_hz, 3000.0)
        elif name in ("xaccel", "x_accel"):
            self.x_axis.accel_hz_s = clamp(value, 100.0, 30000.0)
        elif name in ("yaccel", "y_accel"):
            self.y_axis.accel_hz_s = clamp(value, 100.0, 30000.0)
        elif name in ("xdecel", "x_decel"):
            self.x_axis.decel_hz_s = clamp(value, 100.0, 30000.0)
        elif name in ("ydecel", "y_decel"):
            self.y_axis.decel_hz_s = clamp(value, 100.0, 30000.0)
        elif name in ("laserxreverse", "laser_x_reverse"):
            self.laser_x_reverse = bool(value >= 0.5)
            self.controller.reset()
            self.stop_motion(hard=True)
        elif name in ("laseryreverse", "laser_y_reverse"):
            self.laser_y_reverse = bool(value >= 0.5)
            self.controller.reset()
            self.stop_motion(hard=True)
        elif name in ("xreverse", "x_reverse"):
            self.x_axis.reverse = bool(value >= 0.5)
            self.stop_motion(hard=True)
        elif name in ("yreverse", "y_reverse"):
            self.y_axis.reverse = bool(value >= 0.5)
            self.stop_motion(hard=True)
        elif name in ("laseron", "laser_on"):
            self.laser_output.set(value >= 0.5)
        elif name in ("laseractive", "laseractivelevel"):
            self.laser_output.set_active_level(value)
        elif name in ("clearfault", "reset"):
            self.restart()
        else:
            self.uart.send("[ack,error,unknown_slider,%s]" % name)
            return
        self.uart.send("[ack,slider,%s,%s]" % (name, str(value)))

    def update_fixed_control(self, rect_result, laser_pos, dt):
        if not self.tracking_enabled or self.estop:
            self.stop_motion(hard=False)
            return 4
        now = time.ticks_ms()
        rect_pos = rect_result.get("center") if rect_result else None
        rect_age = rect_result.get("age_ms", 999999) if rect_result else 999999
        laser_age = ticks_diff_ms(now, self.laser.last_ms) if laser_pos is not None else 999999
        tx, ty, state = self.controller.update(rect_pos, rect_age, laser_pos, laser_age, dt, laser_edge_pos=laser_pos)
        if self.laser_x_reverse:
            tx = -tx
        if self.laser_y_reverse:
            ty = -ty
        self.last_x_target = self.x_axis.set_target_hz(tx)
        self.last_y_target = self.y_axis.set_target_hz(ty)
        return state

    def update_ramp(self, dt):
        self.x_axis.update(dt)
        self.y_axis.update(dt)

    def send_plot(self, rect_result, state_code):
        now = time.ticks_ms()
        if ticks_diff_ms(now, self.last_plot_ms) < PLOT_INTERVAL_MS:
            return
        self.last_plot_ms = now
        rect_pos = rect_result.get("center") if rect_result else None
        laser_pos = self.current_laser_pos
        if rect_pos is not None and laser_pos is not None:
            raw_err_x = float(rect_pos[0] - laser_pos[0])
            raw_err_y = float(rect_pos[1] - laser_pos[1])
            rect_x, rect_y = rect_pos
        else:
            # 无效测量只在遥测中显示0，不参与控制器。
            raw_err_x = raw_err_y = 0.0
            rect_x = rect_y = 0.0
        self.uart.send("[plot,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%.2f]" % (
            raw_err_x, raw_err_y, self.x_axis.target_hz, self.y_axis.target_hz,
            self.x_axis.applied_hz, self.y_axis.applied_hz, rect_x, rect_y, int(state_code), self.fps))

    def send_diag(self, rect_result, state_code):
        now = time.ticks_ms()
        if not ENABLE_DIAG_PACKET or ticks_diff_ms(now, self.last_diag_ms) < DIAG_INTERVAL_MS:
            return
        self.last_diag_ms = now
        rect_age = rect_result.get("age_ms", 999999) if rect_result else 999999
        laser_age = ticks_diff_ms(now, self.laser.last_ms) if self.current_laser_pos is not None else 999999
        ex = 0.0 if self.controller.err_x is None else self.controller.err_x
        ey = 0.0 if self.controller.err_y is None else self.controller.err_y
        self.uart.send("[diag,%d,%d,%d,%d,%.2f,%.2f,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f]" % (
            int(rect_age), int(laser_age), int(self.rect.candidate_count), int(self.laser.candidate_count),
            float(self.rect.confidence), float(self.laser.confidence), int(self.rect.miss_frames),
            int(self.laser.ambiguous), int(self.x_axis.limit_hit), int(self.y_axis.limit_hit),
            ex, ey, self.perf_rect_ms, self.perf_laser_ms))

    def print_status(self, rect_result, state_code):
        now = time.ticks_ms()
        if ticks_diff_ms(now, self.last_status_ms) < STATUS_INTERVAL_MS:
            return
        self.last_status_ms = now
        rp = rect_result.get("center") if rect_result else None
        lp = self.current_laser_pos
        if rp is not None and lp is not None:
            print("STAT v10 state=%d err=(%.1f,%.1f) rect=(%.1f,%.1f) laser=(%.1f,%.1f) out=(%.0f,%.0f) cand=%d/%d fps=%.1f" % (
                state_code, rp[0] - lp[0], rp[1] - lp[1], rp[0], rp[1], lp[0], lp[1], self.x_axis.applied_hz, self.y_axis.applied_hz, self.rect.candidate_count, self.laser.candidate_count, self.fps))
        else:
            print("STAT v10 state=%d rect=%d laser=%d out=(%.0f,%.0f) cand=%d/%d fps=%.1f" % (
                state_code, 1 if rp is not None else 0, 1 if lp is not None else 0, self.x_axis.applied_hz, self.y_axis.applied_hz, self.rect.candidate_count, self.laser.candidate_count, self.fps))

    def draw(self, img, rect_result, state_code):
        try:
            if rect_result:
                for c in (rect_result.get("candidates") or [])[:8]:
                    x, y, w, h = c["box_det"]
                    img.draw_rectangle(int(x), int(y), int(w), int(h), color=(60, 120, 255), thickness=1)
                if rect_result.get("box_det") is not None:
                    x, y, w, h = rect_result["box_det"]
                    img.draw_rectangle(int(x), int(y), int(w), int(h), color=(0, 255, 0), thickness=2)
                if rect_result.get("center") is not None:
                    cx = int(round(logical_to_detect_x(rect_result["center"][0])))
                    cy = int(round(logical_to_detect_y(rect_result["center"][1])))
                    img.draw_cross(cx, cy, color=(0, 255, 0), size=7, thickness=2)
            for cand in self.laser.last_candidates_det[:8]:
                img.draw_circle(int(round(cand[0])), int(round(cand[1])), 4, color=(255, 120, 0), thickness=1)
            if self.current_laser_pos is not None:
                lx = int(round(logical_to_detect_x(self.current_laser_pos[0])))
                ly = int(round(logical_to_detect_y(self.current_laser_pos[1])))
                img.draw_cross(lx, ly, color=(255, 0, 0), size=7, thickness=2)
            img.draw_string_advanced(2, 2, 13, "v10 state=%d fps=%.1f out=(%.0f,%.0f)" % (state_code, self.fps, self.x_axis.applied_hz, self.y_axis.applied_hz), color=(255, 255, 255))
        except Exception:
            pass

    def run(self):
        sensor = None
        display_ok = False
        try:
            print("=" * 78)
            print("K230 D36A LASER-RECT TRACKER V10.2 SMOOTH-CONTINUITY")
            print("Closed loop: rectangle center - laser center")
            print("X STEP/DIR GPIO42/26; Y STEP/DIR GPIO43/34; UART3 GPIO32/33; laser GPIO35")
            print("D36A EN1/EN2 must remain tied to D36A board 5V")
            print("Control 50Hz, Ramp 100Hz, no invalid-zero control")
            print("=" * 78)
            try:
                sensor = Sensor(id=SENSOR_ID, width=SENSOR_INPUT_WIDTH, height=SENSOR_INPUT_HEIGHT, fps=SENSOR_FPS)
            except Exception:
                sensor = Sensor()
            sensor.reset()
            sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT, chn=RECT_CHANNEL)
            sensor.set_pixformat(Sensor.RGB888, chn=RECT_CHANNEL)
            sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT, chn=LASER_CHANNEL)
            sensor.set_pixformat(Sensor.RGB565, chn=LASER_CHANNEL)
            if ENABLE_DISPLAY:
                Display.init(Display.ST7701, width=640, height=480, to_ide=DISPLAY_TO_IDE, quality=DISPLAY_QUALITY)
                display_ok = True
            MediaManager.init()
            sensor.run()
            time.sleep_ms(300)
            # 预热双通道
            for _ in range(20):
                os.exitpoint()
                try:
                    sensor.snapshot(chn=RECT_CHANNEL)
                except Exception:
                    pass
                try:
                    sensor.snapshot(chn=LASER_CHANNEL)
                except Exception:
                    pass
                time.sleep_ms(10)
            if LASER_AUTO_ON_AFTER_CAMERA:
                self.laser_output.on()
            self.uart.send("[system,ready,k230_d36a_laser_rect_v10_2_smooth_continuity]")
            self.uart.send("[display,8,8,v10.2 smooth: CH1/2 err CH3/4 targetHz CH5/6 outputHz,14]")
            display_x = int((640 - DETECT_WIDTH) / 2)
            display_y = int((480 - DETECT_HEIGHT) / 2)
            while True:
                os.exitpoint()
                loop_start_us = time.ticks_us()
                now = time.ticks_ms()
                frame_dt = max(0.001, ticks_diff_ms(now, self.last_frame_ms) / 1000.0)
                self.last_frame_ms = now
                inst_fps = 1.0 / frame_dt
                self.fps = inst_fps if self.fps <= 0.1 else 0.85 * self.fps + 0.15 * inst_fps

                cap_start = time.ticks_us()
                img = sensor.snapshot(chn=RECT_CHANNEL)
                img_np = img.to_numpy_ref()
                self.perf_capture_ms = perf_ms(cap_start)

                rect_start = time.ticks_us()
                rect_result = self.rect.step(img, img_np, frame_dt)
                self.perf_rect_ms = perf_ms(rect_start)

                laser_start = time.ticks_us()
                laser_img = sensor.snapshot(chn=LASER_CHANNEL)
                if self.laser_output.enabled:
                    self.current_laser_pos = self.laser.detect(laser_img, rect_result.get("box_det"), frame_dt)
                else:
                    self.laser.reset()
                    self.current_laser_pos = None
                self.perf_laser_ms = perf_ms(laser_start)

                state_code = self.controller.state_code
                # 固定50Hz控制，允许一次循环内补一次，不跟随视觉帧率抖动。
                now = time.ticks_ms()
                if ticks_diff_ms(now, self.last_control_ms) >= CONTROL_PERIOD_MS:
                    cdt = ticks_diff_ms(now, self.last_control_ms) / 1000.0
                    self.last_control_ms = now
                    state_code = self.update_fixed_control(rect_result, self.current_laser_pos, clamp(cdt, 0.008, 0.080))

                # 固定100Hz Ramp，控制指令即使不更新，步进输出仍持续平滑逼近目标。
                now = time.ticks_ms()
                while ticks_diff_ms(now, self.last_ramp_ms) >= RAMP_PERIOD_MS:
                    self.last_ramp_ms = time.ticks_add(self.last_ramp_ms, RAMP_PERIOD_MS)
                    self.update_ramp(RAMP_PERIOD_MS / 1000.0)
                    now = time.ticks_ms()

                self.handle_uart()
                self.send_plot(rect_result, state_code)
                self.send_diag(rect_result, state_code)
                self.print_status(rect_result, state_code)

                self.frame_count += 1
                if display_ok and self.frame_count % max(1, DISPLAY_EVERY_N_FRAMES) == 0:
                    self.draw(img, rect_result, state_code)
                    Display.show_image(img, x=display_x, y=display_y)

                self.perf_total_ms = perf_ms(loop_start_us)
                if self.frame_count % GC_CHECK_INTERVAL_FRAMES == 0:
                    try:
                        if gc.mem_free() < GC_FREE_THRESHOLD:
                            gc.collect()
                    except Exception:
                        gc.collect()
                time.sleep_ms(1)
        except KeyboardInterrupt:
            print("user stop")
        except BaseException as e:
            print("Exception:", e)
            try:
                sys.print_exception(e)
            except Exception:
                pass
        finally:
            self.stop_motion(hard=True)
            self.laser_output.off()
            try:
                self.common_enable.disable()
            except Exception:
                pass
            if isinstance(sensor, Sensor):
                try:
                    sensor.stop()
                except Exception:
                    pass
            if display_ok:
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
            self.x_axis.deinit()
            self.y_axis.deinit()
            self.laser_output.deinit()
            self.common_enable.deinit()
            print("program exited; STEP PWM 0; laser GPIO35 OFF")


if __name__ == "__main__":
    K230D36ALaserRectTrackerV10().run()
