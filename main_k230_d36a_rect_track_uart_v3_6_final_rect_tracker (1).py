# -*- coding: utf-8 -*-
"""
K230 + D36A 二维步进云台黑色矩形视觉追踪 v3.6（最终矩形追踪版）

运行环境：CanMV IDE K230 / MicroPython
视觉基准：main_k230_rect_track_uart_v13_speed_recovery.py
执行器：D36A 双路步进驱动板 + 两相步进电机

功能：
- 320x240 RGB888采集，控制和串口仍使用640x480逻辑坐标；
- cv_lite主检测 + 可自动降级的原生find_rects + Kalman动态ROI + 两级关联；
- 外框/内框嵌套评分、黑框白心对比度、可选亚像素角点精修；
- SEARCH / ACQUIRE / TRACK / COAST / LOST 状态机；
- K230 两路硬件 PWM 直接产生 STEP 脉冲；
- GPIO 控制 X/Y 方向；
- 速度 PID + 速度前馈 + 加减速 Ramp + 换向制动；
- GPIO32/33 UART3 与电脑双向调参；
- 连续发送 10 通道 [plot,...] 数据包；
- 默认直接进入视觉追踪；可通过串口 [motor,test] 手动执行双轴正反转自检。

固定引脚：
    GPIO42 -> D36A STEP1（X轴）
    GPIO26 -> D36A DIR1 （X轴）
    GPIO43 -> D36A STEP2（Y轴）
    GPIO34 -> D36A DIR2 （Y轴）
    GPIO32 -> USB-TTL RX（K230 UART3 TX）
    GPIO33 <- USB-TTL TX（K230 UART3 RX）
    GND    -> D36A GND、USB-TTL GND

D36A 的 EN1、EN2 不接 K230，必须并接到驱动板自身的 5V，保持硬件持续使能。
本版不初始化 GPIO35。严禁把 D36A 的 5V 接到任何 K230 GPIO。

默认 plotMode=0，坐标维持640x480逻辑尺度：
[plot,err_x,err_y,x_target_hz,y_target_hz,x_output_hz,y_output_hz,
 target_x,target_y,state,fps]

state：
0 = SEARCH / LOST
1 = ACQUIRE
2 = TRACK
3 = COAST
4 = ESTOP / TRACKING_DISABLED

重要限制：
- 本系统没有编码器，virtual_steps 只是按输出频率积分得到的软件估计；
- 断电、堵转或失步后，virtual_steps 与真实位置可能不一致；
- 本版不执行自动回中；启动前应把云台手动放在安全中间位置；
- 默认启用相对软限位，目的是降低绕线和撞限位风险。
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
    raise RuntimeError("缺少 cv_lite。请使用当前Yahboom K230固件。")

try:
    import cv2
    from ulab import numpy as np
    CV2_REFINE_AVAILABLE = True
except Exception:
    cv2 = None
    np = None
    CV2_REFINE_AVAILABLE = False

# ============================================================
# 1. 摄像头、显示、串口
# ============================================================
# 视觉在320x240上运行以降低延迟；PID、UART和历史数据仍使用640x480逻辑坐标。
DETECT_WIDTH = 320
DETECT_HEIGHT = 240
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DETECT_TO_LOGICAL_X = CAMERA_WIDTH / float(DETECT_WIDTH)
DETECT_TO_LOGICAL_Y = CAMERA_HEIGHT / float(DETECT_HEIGHT)
IMAGE_SHAPE = [DETECT_HEIGHT, DETECT_WIDTH]

ENABLE_DISPLAY = True
DISPLAY_QUALITY = 45
DISPLAY_EVERY_N_FRAMES = 2

ENABLE_UART = True
UART_BAUD = 115200
UART_TX_PIN = 32
UART_RX_PIN = 33
UART_ID = 3
PLOT_INTERVAL_MS = 20
PLOT_MODE = 0              # 0=控制；1=视觉/Kalman；2=步进驱动诊断
DIAG_INTERVAL_MS = 200
ENABLE_DIAG_PACKET = False
PERF_INTERVAL_MS = 500
ENABLE_PERF_PACKET = True
UART_MAX_PACKETS_PER_LOOP = 6
STATUS_INTERVAL_MS = 1000
SEND_DISPLAY_HELP = True
GC_CHECK_INTERVAL_FRAMES = 90
GC_FREE_THRESHOLD = 190000

SENSOR_ID = 2
SENSOR_INPUT_WIDTH = 1280
SENSOR_INPUT_HEIGHT = 960
SENSOR_FPS = 90

# ============================================================
# 2. D36A STEP/DIR 硬件配置
# ============================================================
X_STEP_PIN = 42            # PWM0
Y_STEP_PIN = 43            # PWM1
X_DIR_PIN = 26
Y_DIR_PIN = 34

# 资料中的方向基准：
# X：DIR=1 顺时针（目标在右时的正方向），DIR=0 逆时针。
# Y：DIR=0 向下（目标在下时的正方向），DIR=1 向上。
X_POSITIVE_DIR_LEVEL = 1
Y_POSITIVE_DIR_LEVEL = 0

# 若实际装配方向相反，只改相应 REVERSE，不要同时改 PID 符号。
X_REVERSE = False
Y_REVERSE = False

# 实机验证：该 D36A 外部 EN 接口为高电平使能。
# EN1、EN2 硬接 D36A 板载 5V；K230 GPIO35 不参与控制。
USE_COMMON_ENABLE_PIN = False
COMMON_ENABLE_PIN = 35
COMMON_ENABLE_ACTIVE_LEVEL = 0
ENABLE_SETTLE_MS = 20

# 可选启动自检。默认关闭，避免每次运行时云台先移动。
# 需要测试时可将 STARTUP_MOTOR_SELF_TEST 改为 True，或串口发送 [motor,test]。
# 自检结束后会自动清零 virtual_steps，再进入视觉追踪。
STARTUP_MOTOR_SELF_TEST = False
SELF_TEST_HZ = 120.0
SELF_TEST_RUN_MS = 800
SELF_TEST_PAUSE_MS = 250
SELF_TEST_LOOP_DT_MS = 10

STEP_DUTY_PERCENT = 50
PWM_INIT_FREQ_HZ = 100
DIR_SETUP_US = 8

# WHEELTEC 例程基于 1.8° 电机、16 细分：200*16=3200 pulse/rev。
MOTOR_FULL_STEPS_PER_REV = 200
MICROSTEP = 16
PULSES_PER_REV = MOTOR_FULL_STEPS_PER_REV * MICROSTEP

# 相对启动位置的软件保护。0 表示关闭；启动前应手动把云台放在中间安全位置。
X_SOFT_LIMIT_STEPS = 3200.0
Y_SOFT_LIMIT_STEPS = 1422.0

# ============================================================
# 3. 矩形检测与目标估计参数（沿用 v13 基线）
# ============================================================
RECT_CFG = {
    # CanMV原生find_rects阈值；值越高越严格。
    "threshold_search": 2800,
    "threshold_track": 2100,
    "threshold_coast": 1600,
    "threshold_periodic_full": 2400,

    # 320x240检测尺度下的基础几何门限。
    "min_area_detect": 120.0,
    "max_area_ratio": 0.46,
    "min_w_detect": 12,
    "min_h_detect": 9,
    "aspect_min": 1.05,
    "aspect_max": 2.38,
    "target_aspect": 1.50,

    # 候选质量分层：同一遍检测结果先关联高置信度，再关联低置信度。
    "quality_high": 0.36,
    "quality_low": 0.18,
    "quality_search_recover": 0.30,
    "max_candidates": 18,

    # cv_lite轮廓检测参数。原生find_rects在本机超过45ms时会自动停用。
    "cv_canny_low": 22,
    "cv_canny_high": 84,
    "cv_approx_epsilon": 0.0205,
    "cv_area_min_ratio": 0.0012,
    "cv_max_angle_cos": 0.40,
    "cv_gaussian_blur_size": 3,
    "native_enable": False,
    "native_slow_ms": 45.0,
    "native_slow_limit": 2,

    # 黑框白心与内外矩形结构。
    "min_contrast": 12.0,
    "require_contrast": False,
    "inner_ratio_min": 0.10,
    "inner_ratio_max": 0.76,
    "inner_ratio_target": 0.43,
    "inner_center_norm_max": 0.23,
    "inner_aspect_delta_max": 0.42,
    "low_stage_min_ring": 0.16,

    # 动态ROI。锁定后主要只在预测窗口内检测。
    "roi_scale_w": 3.30,
    "roi_scale_h": 3.40,
    "roi_min_w": 150,
    "roi_min_h": 120,
    "roi_velocity_lead_s": 0.120,
    "roi_velocity_margin_gain": 0.16,
    "roi_extra_margin": 22,
    "full_scan_interval": 8,
    "full_scan_after_miss": 1,

    # 亚像素角点仅对最终选中候选运行，失败会自动退回原生角点。
    "corner_refine_enable": True,
    "corner_refine_every": 1,
    "corner_refine_window": 4,
    "corner_refine_max_shift": 7.0,
    "corner_refine_pad": 8,
}


TRACK_CFG = {
    "acquire_frames": 2,
    "reacquire_frames": 2,
    "max_coast_frames": 8,
    "keep_track_ms": 5000,
    "control_lead_s": 0.046,

    "gate_tracking_px": 155.0,
    "gate_reacquire_px": 280.0,
    "acquire_jump_px": 95.0,
    "max_area_ratio_tracking": 3.2,
    "max_area_ratio_reacquire": 5.4,
    "max_aspect_delta_tracking": 0.42,
    "max_aspect_delta_reacquire": 0.58,

    # 抗干扰跳变保护：非高质量、非黑白环结构的候选不得从上一帧目标处大幅跳转。
    "jump_guard_px": 95.0,
    "jump_guard_quality": 0.48,
    "jump_guard_ring": 0.18,
    "jump_guard_contrast": 12.0,

    "kalman_accel_noise": 500.0,
    "measurement_var_detect": 10.0,

    # COAST恢复帧与低置信度候选不应立即强校正，避免速度估计尖峰。
    "measurement_var_recover_scale": 1.20,
    "measurement_var_low_scale": 2.20,

    # 限制不可信的Kalman速度，并在连续漏检时逐帧衰减。
    "max_velocity_x_px_s": 560.0,
    "max_velocity_y_px_s": 500.0,
    "coast_velocity_decay": 0.94,

    # 控制中心前瞻位移限制。
    "max_lead_x_px": 92.0,
    "max_lead_y_px": 58.0,

    "initial_pos_var": 90.0,
    "initial_vel_var": 120000.0,
}

# ============================================================
# 4. 步进速度闭环初始参数
# ============================================================
CONTROL_DT_MIN = 0.012
CONTROL_DT_MAX = 0.180
ZERO_CROSS_BRAKE_PX = 12.0
FAST_VEL_START_PX_S = 80.0
FAST_ERR_START_PX = 20.0

PID_CFG = {
    # 输出单位为 STEP Hz，不再是舵机角度增量。
    "x_kp": 5.15,
    "x_ki": 0.000,
    "x_kd": 0.125,
    "x_deadzone_px": 3.0,

    "y_kp": 4.30,
    "y_ki": 0.000,
    "y_kd": 0.120,
    "y_deadzone_px": 4.0,

    "integral_limit": 180.0,
    "derivative_tau_s": 0.090,
}

MOTION_CFG = {
    "x_min_hz": 55.0,
    "y_min_hz": 50.0,
    "x_max_hz": 1000.0,
    "y_max_hz": 820.0,

    "x_accel_hz_s": 3600.0,
    "y_accel_hz_s": 3200.0,
    "x_decel_hz_s": 6500.0,
    "y_decel_hz_s": 5800.0,

    # px/s -> Hz
    "x_ff": 0.28,
    "y_ff": 0.20,
    "x_ff_limit_hz": 300.0,
    "y_ff_limit_hz": 210.0,
    "speed_scale": 1.00,

    # 自适应速度上限：远处允许快，接近中心自动限速，防止高速穿越中心。
    "near_error_px": 18.0,
    "mid_error_px": 55.0,
    "far_error_px": 140.0,
    "x_near_cap_hz": 190.0,
    "y_near_cap_hz": 150.0,
    "x_mid_cap_hz": 540.0,
    "y_mid_cap_hz": 410.0,
    "x_far_cap_hz": 800.0,
    "y_far_cap_hz": 650.0,

    # 目标已经向中心高速接近时，按预计到达时间提前制动。
    "approach_brake_time_s": 0.16,
    "approach_predict_s": 0.070,
    "approach_min_scale": 0.18,
    "approach_max_error_px": 68.0,
    "error_floor_start_px": 24.0,

    # 短时漏检使用 Kalman 预测位置重新计算控制。
    # 普通/向中心运动：保守减速，避免穿越中心后继续冲。
    "coast_enable": True,
    "coast_scale_1": 0.72,
    "coast_scale_2": 0.38,
    "coast_scale_3": 0.15,
    "coast_scale_4": 0.08,
    "coast_scale_5": 0.00,
    "coast_scale_6": 0.00,
    "coast_ff_scale": 0.35,
    "coast_predict_stop_s": 0.080,
    "x_coast_max_hz": 190.0,
    "y_coast_max_hz": 170.0,

    # 目标远离中心且即将离开画面时：
    # 前几帧保持足够速度追向目标，随后逐级衰减，防止长时间盲走。
    "coast_outward_error_px": 42.0,
    "coast_outward_velocity_px_s": 28.0,
    "coast_out_scale_1": 1.00,
    "coast_out_scale_2": 0.92,
    "coast_out_scale_3": 0.78,
    "coast_out_scale_4": 0.58,
    "coast_out_scale_5": 0.36,
    "coast_out_scale_6": 0.18,
    "coast_out_ff_scale": 0.80,
    "x_coast_out_max_hz": 720.0,
    "y_coast_out_max_hz": 580.0,

    # 640x480 图像中，X误差大于240px或Y误差大于180px即接近边缘。
    "x_edge_error_px": 240.0,
    "y_edge_error_px": 180.0,
    "coast_edge_boost": 1.08,

    # COAST只能相对最后一帧有效TRACK指令平滑延续，不允许无检测时自行暴增。
    "coast_reference_gain": 1.18,
    "coast_reference_margin_hz": 45.0,
    "coast_edge_reference_margin_hz": 120.0,
}

# ============================================================
# 5. 状态常量
# ============================================================
STATE_SEARCH = 0
STATE_ACQUIRE = 1
STATE_TRACK = 2
STATE_COAST = 3
STATE_LOST = 4

STATE_NAME = {
    STATE_SEARCH: "SEARCH",
    STATE_ACQUIRE: "ACQUIRE",
    STATE_TRACK: "TRACK",
    STATE_COAST: "COAST",
    STATE_LOST: "LOST",
}


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def ticks_diff_ms(now, old):
    return time.ticks_diff(now, old)


def safe_log_ratio(a, b):
    a = max(1e-6, float(a))
    b = max(1e-6, float(b))
    return abs(math.log(a / b))


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


# ============================================================
# 6. GPIO / EN / STEP PWM 兼容层
# ============================================================
def make_gpio_output(pin_number, initial_value=0):
    """把实际 IO 映射为同号 GPIO，并以输出方式打开。"""
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
    pin = None
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


class StepperAxis:
    """硬件 PWM 输出 STEP，普通 GPIO 输出 DIR。"""

    def __init__(self, name, step_pin, dir_pin, positive_dir_level, reverse,
                 min_hz, max_hz, accel_hz_s, decel_hz_s, soft_limit_steps):
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
        self.output_hz = 0.0       # Ramp 内部有符号频率
        self.applied_hz = 0.0      # 实际送入 PWM 的有符号频率
        self.virtual_steps = 0.0
        self.direction_sign = 0
        self.running = False
        self.last_pwm_freq = 0
        self.limit_hit = 0
        self.enabled = True

        self.dir_pin = make_gpio_output(self.dir_pin_number, self.positive_dir_level)
        try:
            self.pwm = PWM(self.step_pin, freq=int(PWM_INIT_FREQ_HZ), duty=0)
        except TypeError:
            self.pwm = PWM(self.step_pin, int(PWM_INIT_FREQ_HZ), 0)
        self._set_direction(1)
        self._stop_pwm()
        limit_deg = (
            self.soft_limit_steps * 360.0 / float(PULSES_PER_REV)
            if self.soft_limit_steps > 0.0 else 0.0
        )
        print(
            "[%s] STEP GPIO%d / DIR GPIO%d / min=%.0f max=%.0fHz "
            "limit=%.0f steps (±%.1f deg)" % (
                self.name, self.step_pin, self.dir_pin_number,
                self.min_hz, self.max_hz,
                self.soft_limit_steps, limit_deg,
            )
        )

    def _physical_sign(self, logical_sign):
        if self.reverse:
            return -logical_sign
        return logical_sign

    def _set_direction(self, logical_sign):
        logical_sign = 1 if logical_sign >= 0 else -1
        physical_sign = self._physical_sign(logical_sign)
        level = self.positive_dir_level if physical_sign > 0 else (1 - self.positive_dir_level)
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

    def _apply_pwm(self, signed_hz):
        s = sign_of(signed_hz)
        mag = abs(float(signed_hz))
        if s == 0 or mag < self.min_hz or not self.enabled:
            self._stop_pwm()
            return 0.0

        mag = clamp(mag, self.min_hz, self.max_hz)
        self._set_direction(s)
        freq = max(1, int(round(mag)))
        if freq != self.last_pwm_freq:
            self.pwm.freq(freq)
            self.last_pwm_freq = freq
        if not self.running:
            self.pwm.duty(int(STEP_DUTY_PERCENT))
            self.running = True
        self.applied_hz = float(s * freq)
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
        old_applied = self.applied_hz
        target = self._soft_limit_allows(self.target_hz)
        self.target_hz = target

        current = self.output_hz
        cs = sign_of(current)
        ts = sign_of(target)

        # 反向时先按减速度降到 0，再改变 DIR，再从 0 加速。
        if cs != 0 and ts != 0 and cs != ts:
            next_value = move_toward(current, 0.0, self.decel_hz_s * dt)
        elif ts == 0:
            next_value = move_toward(current, 0.0, self.decel_hz_s * dt)
        else:
            speeding_up = abs(target) > abs(current)
            rate = self.accel_hz_s if speeding_up else self.decel_hz_s
            next_value = move_toward(current, target, rate * dt)

        if abs(next_value) < 0.5:
            next_value = 0.0
        next_value = clamp(next_value, -self.max_hz, self.max_hz)
        self.output_hz = self._soft_limit_allows(next_value)
        new_applied = self._apply_pwm(self.output_hz)

        # 软件脉冲累计，仅用于诊断和相对软限位，不是编码器反馈。
        self.virtual_steps += 0.5 * (old_applied + new_applied) * dt
        if self.soft_limit_steps > 0.0:
            self.virtual_steps = clamp(
                self.virtual_steps,
                -self.soft_limit_steps - 1.0,
                self.soft_limit_steps + 1.0,
            )
        return self.applied_hz

    def hard_stop(self):
        self.target_hz = 0.0
        self.output_hz = 0.0
        self._stop_pwm()

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if not self.enabled:
            self.hard_stop()

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
# 7. 速度 PID（输出单位 Hz）
# ============================================================
class VelocityPID:
    def __init__(self, kp, ki, kd, output_limit, integral_limit, derivative_tau):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = float(output_limit)
        self.integral_limit = float(integral_limit)
        self.derivative_tau = float(derivative_tau)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative = 0.0
        self.has_prev = False

    def update(self, error, dt):
        error = float(error)
        dt = max(0.0001, float(dt))

        candidate_integral = clamp(
            self.integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )

        raw_derivative = 0.0
        if self.has_prev:
            raw_derivative = (error - self.prev_error) / dt
        alpha = dt / max(dt, self.derivative_tau + dt)
        self.derivative += alpha * (raw_derivative - self.derivative)

        raw = self.kp * error + self.ki * candidate_integral + self.kd * self.derivative
        out = clamp(raw, -self.output_limit, self.output_limit)

        # 饱和且误差继续把输出推向饱和方向时，不继续积分。
        if abs(raw) <= self.output_limit or error * raw <= 0.0:
            self.integral = candidate_integral

        self.prev_error = error
        self.has_prev = True
        return out


class Kalman1D:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = 0.0
        self.vel = 0.0
        self.p00 = 1.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 1.0
        self.initialized = False

    def initialize(self, pos, vel=0.0):
        self.pos = float(pos)
        self.vel = float(vel)
        self.p00 = TRACK_CFG["initial_pos_var"]
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = TRACK_CFG["initial_vel_var"]
        self.initialized = True

    def predict(self, dt):
        if not self.initialized:
            return
        dt = max(0.0001, float(dt))
        self.pos += self.vel * dt

        # F P F^T
        p00 = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11
        p01 = self.p01 + dt * self.p11
        p10 = self.p10 + dt * self.p11
        p11 = self.p11

        accel = TRACK_CFG["kalman_accel_noise"]
        q = accel * accel
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        p00 += q * dt4 / 4.0
        p01 += q * dt3 / 2.0
        p10 += q * dt3 / 2.0
        p11 += q * dt2

        self.p00 = p00
        self.p01 = p01
        self.p10 = p10
        self.p11 = p11

    def innovation_distance(self, measurement, measurement_var):
        if not self.initialized:
            return 0.0
        residual = float(measurement) - self.pos
        s = max(1e-6, self.p00 + float(measurement_var))
        return residual * residual / s

    def update(self, measurement, measurement_var):
        measurement = float(measurement)
        if not self.initialized:
            self.initialize(measurement)
            return

        r = float(measurement_var)
        residual = measurement - self.pos
        s = max(1e-6, self.p00 + r)
        k0 = self.p00 / s
        k1 = self.p10 / s

        old_p00 = self.p00
        old_p01 = self.p01
        old_p10 = self.p10
        old_p11 = self.p11

        self.pos += k0 * residual
        self.vel += k1 * residual

        self.p00 = (1.0 - k0) * old_p00
        self.p01 = (1.0 - k0) * old_p01
        self.p10 = old_p10 - k1 * old_p00
        self.p11 = old_p11 - k1 * old_p01

        # 抑制数值非对称与负方差。
        cross = 0.5 * (self.p01 + self.p10)
        self.p01 = cross
        self.p10 = cross
        self.p00 = max(1e-5, self.p00)
        self.p11 = max(1e-5, self.p11)


class Kalman2D:
    def __init__(self):
        self.x = Kalman1D()
        self.y = Kalman1D()

    def reset(self):
        self.x.reset()
        self.y.reset()

    def initialize(self, x, y):
        self.x.initialize(x)
        self.y.initialize(y)

    def predict(self, dt):
        self.x.predict(dt)
        self.y.predict(dt)

    def update(self, x, y, measurement_var):
        self.x.update(x, measurement_var)
        self.y.update(y, measurement_var)

    def maha(self, x, y, measurement_var):
        return self.x.innovation_distance(x, measurement_var) + self.y.innovation_distance(y, measurement_var)

    def initialized(self):
        return self.x.initialized and self.y.initialized

    def position(self):
        return self.x.pos, self.y.pos

    def velocity(self):
        return self.x.vel, self.y.vel


# ============================================================
# 9. 矩形检测与目标关联
# ============================================================
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


def pixel_gray(img_np, x, y):
    x = int(clamp(x, 0, DETECT_WIDTH - 1))
    y = int(clamp(y, 0, DETECT_HEIGHT - 1))
    try:
        channels = int(img_np.shape[2])
        if channels == 2:
            # Sensor.RGB565 的 numpy 引用为 [low_byte, high_byte]。
            value = int(img_np[y, x, 0]) | (int(img_np[y, x, 1]) << 8)
            r = ((value >> 11) & 0x1F) * 255.0 / 31.0
            g = ((value >> 5) & 0x3F) * 255.0 / 63.0
            b = (value & 0x1F) * 255.0 / 31.0
            return (r * 30.0 + g * 59.0 + b * 11.0) / 100.0
        c0 = int(img_np[y, x, 0])
        c1 = int(img_np[y, x, 1])
        c2 = int(img_np[y, x, 2])
        return (c0 * 30 + c1 * 59 + c2 * 11) / 100.0
    except Exception:
        return 0.0


def ring_contrast(img_np, x, y, w, h):
    """粗略估计黑框白心对比度：中心亮度减去四边亮度。"""
    if w < 8 or h < 8:
        return -255.0

    inner_vals = []
    border_vals = []
    for fy in (0.38, 0.50, 0.62):
        for fx in (0.38, 0.50, 0.62):
            inner_vals.append(pixel_gray(img_np, x + w * fx, y + h * fy))

    for t in (0.14, 0.32, 0.50, 0.68, 0.86):
        border_vals.append(pixel_gray(img_np, x + w * t, y + h * 0.08))
        border_vals.append(pixel_gray(img_np, x + w * t, y + h * 0.92))
        border_vals.append(pixel_gray(img_np, x + w * 0.08, y + h * t))
        border_vals.append(pixel_gray(img_np, x + w * 0.92, y + h * t))

    return (
        sum(inner_vals) / max(1, len(inner_vals))
        - sum(border_vals) / max(1, len(border_vals))
    )


def order_corners(corners):
    """按质心极角排序四角，便于计算两条对角线交点。"""
    if corners is None or len(corners) != 4:
        return corners
    cx = sum(float(p[0]) for p in corners) / 4.0
    cy = sum(float(p[1]) for p in corners) / 4.0
    ordered = list(corners)
    ordered.sort(key=lambda p: math.atan2(float(p[1]) - cy, float(p[0]) - cx))
    return ordered


def line_intersection(p1, p2, p3, p4):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None

    n1 = x1 * y2 - y1 * x2
    n2 = x3 * y4 - y3 * x4
    px = (n1 * (x3 - x4) - (x1 - x2) * n2) / den
    py = (n1 * (y3 - y4) - (y1 - y2) * n2) / den
    return px, py


def quadrilateral_center(corners):
    ordered = order_corners(corners)
    if ordered is None or len(ordered) != 4:
        return 0.0, 0.0
    cross = line_intersection(ordered[0], ordered[2], ordered[1], ordered[3])
    if cross is not None:
        return cross
    return (
        sum(float(p[0]) for p in ordered) / 4.0,
        sum(float(p[1]) for p in ordered) / 4.0,
    )


def box_from_corners(corners, pad=1.5):
    ordered = order_corners(corners)
    xs = [float(p[0]) for p in ordered]
    ys = [float(p[1]) for p in ordered]
    x0 = clamp(min(xs) - pad, 0.0, DETECT_WIDTH - 1.0)
    y0 = clamp(min(ys) - pad, 0.0, DETECT_HEIGHT - 1.0)
    x1 = clamp(max(xs) + pad, x0 + 2.0, DETECT_WIDTH)
    y1 = clamp(max(ys) + pad, y0 + 2.0, DETECT_HEIGHT)
    return (
        int(round(x0)), int(round(y0)),
        int(round(x1 - x0)), int(round(y1 - y0)),
    )


def candidate_quality(candidate):
    aspect_error = (
        abs(candidate["aspect"] - RECT_CFG["target_aspect"])
        / max(0.25, RECT_CFG["target_aspect"])
    )
    aspect_score = clamp(1.0 - aspect_error, 0.0, 1.0)
    area_score = clamp(candidate["area_detect"] / 1800.0, 0.0, 1.0)
    magnitude_score = clamp(candidate["magnitude"] / 26000.0, 0.0, 1.0)
    contrast_score = clamp(
        (candidate["contrast"] - RECT_CFG["min_contrast"] + 12.0) / 46.0,
        0.0, 1.0,
    )
    ring_score = float(candidate.get("ring_score", 0.0))
    # v3.4：降低面积/幅值对大背景矩形的偏好，强化黑白边框与内外矩形结构。
    return clamp(
        0.32 * aspect_score
        + 0.11 * area_score
        + 0.12 * magnitude_score
        + 0.25 * contrast_score
        + 0.18 * ring_score
        + 0.08,
        0.0, 1.0,
    )


class RectangleTracker:
    """v3.4：基于v3.3大角度范围，收紧抗干扰、用角点多边形修正框选。"""

    def __init__(self):
        self.kalman = Kalman2D()
        self.state = STATE_SEARCH
        self.acquire_count = 0
        self.pending = None
        self.last_candidate = None
        self.last_area = None
        self.last_aspect = None
        self.last_box = None
        self.last_box_det = None
        self.last_corners_det = None
        self.miss_frames = 0
        self.last_visual_ms = time.ticks_ms()
        self.raw_center = None
        self.filtered_center = None
        self.last_confidence = 0.0
        self.detector_miss_count = 0
        self.strict_miss_count = 0
        self.relaxed_recover_count = 0
        self.hard_miss_count = 0
        self.last_relaxed_used = 0
        self.relaxed_skipped_count = 0
        self.ever_locked = False
        self.frame_index = 0
        self.last_roi = (0, 0, DETECT_WIDTH, DETECT_HEIGHT)
        self.last_full_scan = 1
        self.corner_refine_fail_count = 0
        self.refine_runtime_enabled = True
        self.low_stage_count = 0
        self.native_runtime_enabled = bool(RECT_CFG["native_enable"])
        self.native_slow_count = 0
        self.last_native_ms = 0.0
        self.last_detector_mode = "CV"

    def reset(self):
        self.kalman.reset()
        self.state = STATE_SEARCH
        self.acquire_count = 0
        self.pending = None
        self.last_candidate = None
        self.last_area = None
        self.last_aspect = None
        self.last_box = None
        self.last_box_det = None
        self.last_corners_det = None
        self.miss_frames = 0
        self.last_visual_ms = time.ticks_ms()
        self.raw_center = None
        self.filtered_center = None
        self.last_confidence = 0.0
        self.detector_miss_count = 0
        self.strict_miss_count = 0
        self.relaxed_recover_count = 0
        self.hard_miss_count = 0
        self.last_relaxed_used = 0
        self.relaxed_skipped_count = 0
        self.ever_locked = False
        self.frame_index = 0
        self.last_roi = (0, 0, DETECT_WIDTH, DETECT_HEIGHT)
        self.last_full_scan = 1
        self.corner_refine_fail_count = 0
        self.refine_runtime_enabled = True
        self.low_stage_count = 0
        self.native_runtime_enabled = bool(RECT_CFG["native_enable"])
        self.native_slow_count = 0
        self.last_native_ms = 0.0
        self.last_detector_mode = "CV"

    @staticmethod
    def clamp_roi(x, y, w, h):
        w = int(clamp(round(w), 32, DETECT_WIDTH))
        h = int(clamp(round(h), 32, DETECT_HEIGHT))
        x = int(clamp(round(x), 0, DETECT_WIDTH - w))
        y = int(clamp(round(y), 0, DETECT_HEIGHT - h))
        return x, y, w, h

    def build_roi(self):
        full_scan = (
            self.state in (STATE_SEARCH, STATE_LOST)
            or not self.kalman.initialized()
            or self.last_box_det is None
            or self.miss_frames >= int(RECT_CFG["full_scan_after_miss"])
            or (self.frame_index % max(1, int(RECT_CFG["full_scan_interval"]))) == 0
        )
        if full_scan:
            return (0, 0, DETECT_WIDTH, DETECT_HEIGHT), True

        px, py = self.kalman.position()
        vx, vy = self.kalman.velocity()
        px = logical_to_detect_x(px)
        py = logical_to_detect_y(py)
        vx = vx / DETECT_TO_LOGICAL_X
        vy = vy / DETECT_TO_LOGICAL_Y

        _, _, bw, bh = self.last_box_det
        roi_w = max(
            float(RECT_CFG["roi_min_w"]),
            float(bw) * RECT_CFG["roi_scale_w"],
        )
        roi_h = max(
            float(RECT_CFG["roi_min_h"]),
            float(bh) * RECT_CFG["roi_scale_h"],
        )

        lead_s = RECT_CFG["roi_velocity_lead_s"]
        center_x = px + vx * lead_s
        center_y = py + vy * lead_s
        roi_w += (
            abs(vx) * RECT_CFG["roi_velocity_margin_gain"]
            + 2.0 * RECT_CFG["roi_extra_margin"]
        )
        roi_h += (
            abs(vy) * RECT_CFG["roi_velocity_margin_gain"]
            + 2.0 * RECT_CFG["roi_extra_margin"]
        )
        return self.clamp_roi(
            center_x - roi_w * 0.5,
            center_y - roi_h * 0.5,
            roi_w, roi_h,
        ), False

    def threshold_for_state(self, full_scan):
        if self.state == STATE_COAST:
            return int(RECT_CFG["threshold_coast"])
        if self.state == STATE_TRACK and not full_scan:
            return int(RECT_CFG["threshold_track"])
        if self.state == STATE_TRACK and full_scan:
            return int(RECT_CFG["threshold_periodic_full"])
        return int(RECT_CFG["threshold_search"])

    def raw_candidate_from_rect(self, rect_obj, img_np):
        x, y, w, h = [int(v) for v in rect_obj.rect()]
        if w <= 0 or h <= 0:
            return None

        area_detect = float(w * h)
        max_area = DETECT_WIDTH * DETECT_HEIGHT * RECT_CFG["max_area_ratio"]
        if area_detect < RECT_CFG["min_area_detect"] or area_detect > max_area:
            return None

        long_side, short_side = max(w, h), min(w, h)
        if (
            long_side < RECT_CFG["min_w_detect"]
            or short_side < RECT_CFG["min_h_detect"]
        ):
            return None

        aspect = float(long_side) / max(1.0, float(short_side))
        if not (RECT_CFG["aspect_min"] <= aspect <= RECT_CFG["aspect_max"]):
            return None

        corners_det = [(float(p[0]), float(p[1])) for p in rect_obj.corners()]
        corners_det = order_corners(corners_det)
        center_det = quadrilateral_center(corners_det)
        x, y, w, h = box_from_corners(corners_det)
        area_detect = float(w * h)
        magnitude = float(rect_obj.magnitude())

        c = {
            "x_det": x, "y_det": y, "w_det": w, "h_det": h,
            "box_det": (x, y, w, h),
            "corners_det": corners_det,
            "center_det": center_det,
            "area_detect": area_detect,
            "aspect": aspect,
            "contrast": 0.0,
            "magnitude": magnitude,
            "ring_score": 0.0,
            "inner_ratio": 0.0,
            "inner_box_det": None,
            "low_stage": False,
            "refined": False,
        }
        self.update_logical_geometry(c)
        return c

    def raw_candidate_from_cv(self, raw, roi_x, roi_y):
        if raw is None or len(raw) < 12:
            return None
        x = int(raw[0]) + int(roi_x)
        y = int(raw[1]) + int(roi_y)
        w = int(raw[2])
        h = int(raw[3])
        if w <= 0 or h <= 0:
            return None

        area_detect = float(w * h)
        max_area = DETECT_WIDTH * DETECT_HEIGHT * RECT_CFG["max_area_ratio"]
        if area_detect < RECT_CFG["min_area_detect"] or area_detect > max_area:
            return None

        long_side, short_side = max(w, h), min(w, h)
        if (
            long_side < RECT_CFG["min_w_detect"]
            or short_side < RECT_CFG["min_h_detect"]
        ):
            return None

        aspect = float(long_side) / max(1.0, float(short_side))
        if not (RECT_CFG["aspect_min"] <= aspect <= RECT_CFG["aspect_max"]):
            return None

        corners_det = (
            (float(raw[4]) + roi_x, float(raw[5]) + roi_y),
            (float(raw[6]) + roi_x, float(raw[7]) + roi_y),
            (float(raw[8]) + roi_x, float(raw[9]) + roi_y),
            (float(raw[10]) + roi_x, float(raw[11]) + roi_y),
        )
        corners_det = order_corners(corners_det)
        center_det = quadrilateral_center(corners_det)
        x, y, w, h = box_from_corners(corners_det)
        area_detect = float(w * h)
        # cv_lite没有原生magnitude，使用面积与轮廓尺寸生成稳定代理值。
        magnitude = min(26000.0, area_detect * 10.0 + (w + h) * 45.0)

        c = {
            "x_det": x, "y_det": y, "w_det": w, "h_det": h,
            "box_det": (x, y, w, h),
            "corners_det": corners_det,
            "center_det": center_det,
            "area_detect": area_detect,
            "aspect": aspect,
            "contrast": 0.0,
            "magnitude": magnitude,
            "ring_score": 0.0,
            "inner_ratio": 0.0,
            "inner_box_det": None,
            "low_stage": False,
            "refined": False,
            "detector": "CV",
        }
        self.update_logical_geometry(c)
        return c

    @staticmethod
    def inner_relation_score(outer, inner):
        ox, oy, ow, oh = outer["box_det"]
        ix, iy, iw, ih = inner["box_det"]
        ocx, ocy = outer["center_det"]
        icx, icy = inner["center_det"]

        if inner["area_detect"] >= outer["area_detect"]:
            return 0.0
        if ix <= ox or iy <= oy or ix + iw >= ox + ow or iy + ih >= oy + oh:
            return 0.0

        ratio = inner["area_detect"] / max(1.0, outer["area_detect"])
        if not (
            RECT_CFG["inner_ratio_min"]
            <= ratio
            <= RECT_CFG["inner_ratio_max"]
        ):
            return 0.0

        diag = math.sqrt(float(ow * ow + oh * oh))
        center_norm = math.sqrt((icx - ocx) ** 2 + (icy - ocy) ** 2) / max(1.0, diag)
        if center_norm > RECT_CFG["inner_center_norm_max"]:
            return 0.0

        aspect_delta = abs(inner["aspect"] - outer["aspect"])
        if aspect_delta > RECT_CFG["inner_aspect_delta_max"]:
            return 0.0

        center_score = clamp(
            1.0 - center_norm / max(0.01, RECT_CFG["inner_center_norm_max"]),
            0.0, 1.0,
        )
        ratio_score = clamp(
            1.0 - abs(ratio - RECT_CFG["inner_ratio_target"]) / 0.34,
            0.0, 1.0,
        )
        aspect_score = clamp(
            1.0 - aspect_delta / max(0.01, RECT_CFG["inner_aspect_delta_max"]),
            0.0, 1.0,
        )
        return 0.46 * center_score + 0.34 * ratio_score + 0.20 * aspect_score

    def attach_inner_ring_scores(self, candidates):
        for outer in candidates:
            best_inner = None
            best_score = 0.0
            for inner in candidates:
                if inner is outer:
                    continue
                score = self.inner_relation_score(outer, inner)
                if score > best_score:
                    best_score = score
                    best_inner = inner
            outer["ring_score"] = best_score
            if best_inner is not None:
                outer["inner_ratio"] = (
                    best_inner["area_detect"] / max(1.0, outer["area_detect"])
                )
                outer["inner_box_det"] = best_inner["box_det"]
            outer["quality"] = candidate_quality(outer)

    def update_logical_geometry(self, candidate):
        cx_det, cy_det = candidate["center_det"]
        x_det, y_det, w_det, h_det = candidate["box_det"]
        candidate["center"] = (
            detect_to_logical_x(cx_det),
            detect_to_logical_y(cy_det),
        )
        candidate["x"] = detect_to_logical_x(x_det)
        candidate["y"] = detect_to_logical_y(y_det)
        candidate["w"] = detect_to_logical_x(w_det)
        candidate["h"] = detect_to_logical_y(h_det)
        candidate["area"] = candidate["w"] * candidate["h"]
        candidate["corners"] = tuple(
            (
                detect_to_logical_x(p[0]),
                detect_to_logical_y(p[1]),
            )
            for p in candidate["corners_det"]
        )

    def generate_candidates(self, img, img_np, roi):
        roi_x, roi_y, roi_w, roi_h = [int(v) for v in roi]
        candidates = []

        # 原生find_rects在当前固件实测全图约80ms。保留兼容路径，
        # 但连续两次超过阈值后自动切换为已验证可用的cv_lite。
        if self.native_runtime_enabled:
            native_start_us = time.ticks_us()
            try:
                raw_rects = img.find_rects(
                    roi=roi,
                    threshold=self.threshold_for_state(self.last_full_scan != 0),
                )
                self.last_native_ms = perf_ms(native_start_us)
                if self.last_native_ms > RECT_CFG["native_slow_ms"]:
                    self.native_slow_count += 1
                else:
                    self.native_slow_count = max(0, self.native_slow_count - 1)

                for r in raw_rects:
                    c = self.raw_candidate_from_rect(r, img_np)
                    if c is not None:
                        c["detector"] = "NATIVE"
                        candidates.append(c)
            except Exception as e:
                self.last_native_ms = perf_ms(native_start_us)
                self.native_slow_count = int(RECT_CFG["native_slow_limit"])
                print("native find_rects unavailable, switch to cv_lite:", e)

            if self.native_slow_count >= int(RECT_CFG["native_slow_limit"]):
                self.native_runtime_enabled = False
                print(
                    "native find_rects disabled: %.1fms, cv_lite ROI enabled"
                    % self.last_native_ms
                )

        # cv_lite是本机v2.x已经验证可运行的主检测器。
        # 原生没有候选时同帧补检；原生停用后只运行cv_lite。
        if (not candidates) or (not self.native_runtime_enabled):
            try:
                # cv_lite使用连续的完整RGB888帧，避免ulab ROI切片的步长兼容问题。
                # 检测后再按动态ROI筛选候选；320x240全图开销仍远低于旧版640x480。
                raw_cv = cv_lite.rgb888_find_rectangles_with_corners(
                    IMAGE_SHAPE, img_np,
                    int(RECT_CFG["cv_canny_low"]),
                    int(RECT_CFG["cv_canny_high"]),
                    float(RECT_CFG["cv_approx_epsilon"]),
                    float(RECT_CFG["cv_area_min_ratio"]),
                    float(RECT_CFG["cv_max_angle_cos"]),
                    int(RECT_CFG["cv_gaussian_blur_size"]),
                )
                roi_x1 = roi_x + roi_w
                roi_y1 = roi_y + roi_h
                for raw in raw_cv:
                    c = self.raw_candidate_from_cv(raw, 0, 0)
                    if c is None:
                        continue
                    ccx, ccy = c["center_det"]
                    if (
                        self.last_full_scan
                        or (
                            roi_x <= ccx <= roi_x1
                            and roi_y <= ccy <= roi_y1
                        )
                    ):
                        candidates.append(c)
                self.last_detector_mode = "CV"
            except Exception as e:
                print("cv_lite detector error:", e)
        else:
            self.last_detector_mode = "NATIVE"

        # 去除同位置重复候选，避免原生与cv_lite同时返回同一矩形。
        dedup = []
        for c in candidates:
            keep = True
            cx, cy = c["center_det"]
            for d in dedup:
                dx, dy = d["center_det"]
                if (
                    abs(cx - dx) < 5.0
                    and abs(cy - dy) < 5.0
                    and abs(c["area_detect"] - d["area_detect"])
                    < 0.25 * max(c["area_detect"], d["area_detect"])
                ):
                    if c["magnitude"] > d["magnitude"]:
                        dedup.remove(d)
                        break
                    keep = False
                    break
            if keep:
                dedup.append(c)
        candidates = dedup

        # 先保留幅值/面积较强者，再进行O(n²)内外矩形配对。
        candidates.sort(
            key=lambda c: (
                c["magnitude"]
                + 3.0 * c["area_detect"]
            ),
            reverse=True,
        )
        candidates = candidates[:int(RECT_CFG["max_candidates"])]
        for c in candidates:
            c["contrast"] = ring_contrast(
                img_np, c["x_det"], c["y_det"], c["w_det"], c["h_det"]
            )
        self.attach_inner_ring_scores(candidates)
        candidates.sort(key=lambda c: c["quality"], reverse=True)
        return candidates

    def search_best(self, candidates):
        min_quality = (
            RECT_CFG["quality_search_recover"]
            if self.ever_locked else RECT_CFG["quality_high"]
        )
        best = None
        best_score = -1e9
        for c in candidates:
            if c["quality"] < min_quality:
                continue
            if (
                RECT_CFG["require_contrast"]
                and c["contrast"] < RECT_CFG["min_contrast"]
            ):
                continue
            if (
                self.ever_locked
                and c["ring_score"] < RECT_CFG["low_stage_min_ring"]
                and c["contrast"] < RECT_CFG["min_contrast"]
            ):
                continue
            score = (
                c["quality"]
                + 0.28 * c["ring_score"]
                + (0.14 if c["contrast"] >= RECT_CFG["min_contrast"] else 0.0)
            )
            if score > best_score:
                best = c
                best_score = score
        return best

    def associate(self, candidates, reacquire=False, low_stage=False):
        if not self.kalman.initialized():
            return self.search_best(candidates)

        px, py = self.kalman.position()
        gate_px = (
            TRACK_CFG["gate_reacquire_px"]
            if reacquire else TRACK_CFG["gate_tracking_px"]
        )
        if low_stage:
            # v3.4：低置信度候选必须更接近预测轨迹，避免背景矩形接管目标。
            gate_px *= 0.78

        max_area_ratio = (
            TRACK_CFG["max_area_ratio_reacquire"]
            if reacquire else TRACK_CFG["max_area_ratio_tracking"]
        )
        max_aspect_delta = (
            TRACK_CFG["max_aspect_delta_reacquire"]
            if reacquire else TRACK_CFG["max_aspect_delta_tracking"]
        )
        if low_stage:
            max_area_ratio = min(max_area_ratio, 2.8)
            max_aspect_delta = min(max_aspect_delta, 0.38)

        best = None
        best_cost = 1e12
        for c in candidates:
            quality = float(c.get("quality", 0.0))
            if low_stage:
                if quality < RECT_CFG["quality_low"]:
                    continue
                if (
                    c["ring_score"] < RECT_CFG["low_stage_min_ring"]
                    and c["contrast"] < RECT_CFG["min_contrast"]
                ):
                    continue
            elif quality < RECT_CFG["quality_high"]:
                continue

            cx, cy = c["center"]
            dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist > gate_px:
                continue

            if self.last_candidate is not None:
                lx, ly = self.last_candidate.get("center", (px, py))
                jump = math.sqrt((cx - lx) ** 2 + (cy - ly) ** 2)
                if jump > TRACK_CFG["jump_guard_px"]:
                    structurally_valid = (
                        c.get("ring_score", 0.0) >= TRACK_CFG["jump_guard_ring"]
                        or c.get("contrast", 0.0) >= TRACK_CFG["jump_guard_contrast"]
                    )
                    if (quality < TRACK_CFG["jump_guard_quality"]) or (not structurally_valid):
                        continue

            area_penalty = 0.0
            if self.last_area is not None:
                ratio = max(c["area"], self.last_area) / max(
                    1.0, min(c["area"], self.last_area)
                )
                if ratio > max_area_ratio:
                    continue
                area_penalty = safe_log_ratio(c["area"], self.last_area)

            aspect_penalty = 0.0
            if self.last_aspect is not None:
                aspect_penalty = abs(c["aspect"] - self.last_aspect)
                if aspect_penalty > max_aspect_delta:
                    continue

            cost = (
                0.028 * dist
                + 2.2 * area_penalty
                + 2.5 * aspect_penalty
                + 2.8 * (1.0 - quality)
                - 1.10 * c["ring_score"]
            )
            if c["contrast"] < RECT_CFG["min_contrast"]:
                cost += 0.80
            if low_stage:
                cost += 0.55

            if cost < best_cost:
                best = c
                best_cost = cost
        return best

    def acquire_consistent(self, candidate):
        if self.pending is None:
            return True
        ax, ay = self.pending["center"]
        bx, by = candidate["center"]
        jump = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        area_ratio = max(candidate["area"], self.pending["area"]) / max(
            1.0, min(candidate["area"], self.pending["area"])
        )
        aspect_delta = abs(candidate["aspect"] - self.pending["aspect"])
        return (
            jump <= TRACK_CFG["acquire_jump_px"]
            and area_ratio <= 2.8
            and aspect_delta <= 0.42
        )

    def refine_candidate(self, img_np, candidate, force=False):
        if (
            not CV2_REFINE_AVAILABLE
            or not self.refine_runtime_enabled
            or not RECT_CFG["corner_refine_enable"]
            or candidate is None
        ):
            return 0.0

        interval = max(1, int(RECT_CFG["corner_refine_every"]))
        if not force and (self.frame_index % interval) != 0:
            return 0.0

        start_us = time.ticks_us()
        try:
            x, y, w, h = candidate["box_det"]
            pad = int(RECT_CFG["corner_refine_pad"])
            x0 = int(clamp(x - pad, 0, DETECT_WIDTH - 1))
            y0 = int(clamp(y - pad, 0, DETECT_HEIGHT - 1))
            x1 = int(clamp(x + w + pad, x0 + 8, DETECT_WIDTH))
            y1 = int(clamp(y + h + pad, y0 + 8, DETECT_HEIGHT))
            patch = img_np[y0:y1, x0:x1]
            if int(patch.shape[2]) == 2:
                if hasattr(cv2, "COLOR_BGR5652GRAY"):
                    gray = cv2.cvtColor(patch, cv2.COLOR_BGR5652GRAY)
                elif hasattr(cv2, "COLOR_RGB5652GRAY"):
                    gray = cv2.cvtColor(patch, cv2.COLOR_RGB5652GRAY)
                else:
                    raise RuntimeError("cv2 RGB565-to-gray conversion unavailable")
            else:
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            point_data = []
            for px, py in candidate["corners_det"]:
                point_data.append([[float(px - x0), float(py - y0)]])
            points = np.array(point_data, dtype=np.float)

            win = int(RECT_CFG["corner_refine_window"])
            refined = cv2.cornerSubPix(
                gray, points, (win, win), (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    12, 0.03,
                ),
            )
            new_corners = []
            max_shift = float(RECT_CFG["corner_refine_max_shift"])
            for i in range(4):
                nx = float(refined[i, 0, 0]) + x0
                ny = float(refined[i, 0, 1]) + y0
                ox, oy = candidate["corners_det"][i]
                if math.sqrt((nx - ox) ** 2 + (ny - oy) ** 2) > max_shift:
                    raise ValueError("cornerSubPix shift too large")
                new_corners.append((nx, ny))

            new_corners = order_corners(new_corners)
            candidate["corners_det"] = new_corners
            candidate["center_det"] = quadrilateral_center(new_corners)
            x, y, w, h = box_from_corners(new_corners)
            candidate["x_det"] = x
            candidate["y_det"] = y
            candidate["w_det"] = w
            candidate["h_det"] = h
            candidate["box_det"] = (x, y, w, h)
            candidate["area_detect"] = float(w * h)
            candidate["refined"] = True
            self.update_logical_geometry(candidate)
        except Exception as e:
            self.corner_refine_fail_count += 1
            if self.corner_refine_fail_count >= 3:
                self.refine_runtime_enabled = False
                print("cornerSubPix disabled after repeated failure:", e)
        return perf_ms(start_us)

    def measurement_var_for(self, candidate, acquire=False, recovering=False):
        quality = float(candidate.get("quality", 0.5))
        area = max(1.0, float(candidate.get("area", 1000.0)))
        quality_scale = clamp(1.45 - 0.78 * quality, 0.65, 1.30)
        area_scale = clamp(math.sqrt(1600.0 / area), 0.72, 1.40)
        acquire_scale = 1.20 if acquire else 1.0
        recover_scale = (
            TRACK_CFG["measurement_var_recover_scale"] if recovering else 1.0
        )
        low_scale = (
            TRACK_CFG["measurement_var_low_scale"]
            if candidate.get("low_stage", False) else 1.0
        )
        refined_scale = 0.78 if candidate.get("refined", False) else 1.0
        return (
            TRACK_CFG["measurement_var_detect"]
            * quality_scale * area_scale
            * acquire_scale * recover_scale
            * low_scale * refined_scale
        )

    def clamp_kalman_velocity(self):
        self.kalman.x.vel = clamp(
            self.kalman.x.vel,
            -TRACK_CFG["max_velocity_x_px_s"],
            TRACK_CFG["max_velocity_x_px_s"],
        )
        self.kalman.y.vel = clamp(
            self.kalman.y.vel,
            -TRACK_CFG["max_velocity_y_px_s"],
            TRACK_CFG["max_velocity_y_px_s"],
        )

    def accept_candidate(self, candidate, acquire=False, recovering=False):
        cx, cy = candidate["center"]
        self.raw_center = (cx, cy)
        if not self.kalman.initialized():
            self.kalman.initialize(cx, cy)
        else:
            self.kalman.update(
                cx, cy,
                self.measurement_var_for(candidate, acquire, recovering),
            )
        self.clamp_kalman_velocity()
        self.last_candidate = candidate
        self.last_area = candidate["area"]
        self.last_aspect = candidate["aspect"]
        self.last_box = (
            candidate["x"], candidate["y"],
            candidate["w"], candidate["h"],
        )
        self.last_box_det = candidate["box_det"]
        self.last_corners_det = candidate["corners_det"]
        self.last_visual_ms = time.ticks_ms()
        self.miss_frames = 0
        self.last_confidence = float(candidate["quality"])

    def control_center(self):
        x, y = self.kalman.position()
        vx, vy = self.kalman.velocity()
        lead = TRACK_CFG["control_lead_s"]
        x += clamp(
            vx * lead,
            -TRACK_CFG["max_lead_x_px"],
            TRACK_CFG["max_lead_x_px"],
        )
        y += clamp(
            vy * lead,
            -TRACK_CFG["max_lead_y_px"],
            TRACK_CFG["max_lead_y_px"],
        )
        self.filtered_center = (x, y)
        return (
            int(round(clamp(x, 0, CAMERA_WIDTH - 1))),
            int(round(clamp(y, 0, CAMERA_HEIGHT - 1))),
        )

    def step(self, img, img_np, dt):
        step_start_us = time.ticks_us()
        self.frame_index += 1
        if self.kalman.initialized():
            self.kalman.predict(dt)

        roi, full_scan = self.build_roi()
        self.last_roi = roi
        self.last_full_scan = 1 if full_scan else 0

        detect_start_us = time.ticks_us()
        candidates = self.generate_candidates(img, img_np, roi)
        detect_ms = perf_ms(detect_start_us)

        associate_start_us = time.ticks_us()
        reacquire = self.state in (STATE_LOST, STATE_COAST)
        high_floor = RECT_CFG["quality_high"]
        if self.ever_locked and self.state in (STATE_SEARCH, STATE_LOST):
            high_floor = RECT_CFG["quality_search_recover"]
        high_candidates = [
            c for c in candidates
            if c["quality"] >= high_floor
        ]
        low_candidates = [
            c for c in candidates
            if RECT_CFG["quality_low"] <= c["quality"] < RECT_CFG["quality_high"]
        ]

        if self.state == STATE_LOST and self.kalman.initialized():
            # v3.4：丢失后的重捕获必须优先服从上一条轨迹，
            # 不允许全画面高质量背景矩形直接接管目标。
            candidate = self.associate(
                high_candidates, reacquire=True, low_stage=False
            )
            if candidate is None:
                candidate = self.associate(
                    low_candidates, reacquire=True, low_stage=True
                )
        elif self.state in (STATE_SEARCH, STATE_ACQUIRE):
            candidate = self.search_best(high_candidates)
        else:
            candidate = self.associate(
                high_candidates, reacquire=reacquire, low_stage=False
            )
            if candidate is None:
                candidate = self.associate(
                    low_candidates, reacquire=reacquire, low_stage=True
                )

        low_used = 0
        if candidate is not None and candidate in low_candidates:
            candidate["low_stage"] = True
            low_used = 1
            self.low_stage_count += 1

        associate_ms = perf_ms(associate_start_us)
        refine_ms = 0.0
        if candidate is not None:
            refine_ms = self.refine_candidate(
                img_np, candidate,
                force=bool(low_used or self.state in (STATE_COAST, STATE_LOST)),
            )

        source = "NONE"
        confidence = 0.0
        if candidate is not None:
            had_history = self.ever_locked
            if self.state in (STATE_SEARCH, STATE_ACQUIRE, STATE_LOST):
                if not self.acquire_consistent(candidate):
                    self.acquire_count = 1
                else:
                    self.acquire_count += 1
                self.pending = candidate
                self.accept_candidate(
                    candidate,
                    acquire=True,
                    recovering=(self.state == STATE_LOST),
                )
                self.state = STATE_ACQUIRE
                source = "DETECT_LOW" if low_used else "DETECT_HI"
                confidence = candidate["quality"]
                required = (
                    TRACK_CFG["reacquire_frames"]
                    if had_history else TRACK_CFG["acquire_frames"]
                )
                if (
                    candidate["quality"] >= 0.58
                    and (
                        candidate.get("ring_score", 0.0) >= 0.18
                        or candidate.get("contrast", 0.0) >= RECT_CFG["min_contrast"]
                    )
                ):
                    required = 1
                if self.acquire_count >= required:
                    self.state = STATE_TRACK
                    self.ever_locked = True
                    self.pending = None
            else:
                recovering = self.state == STATE_COAST
                self.accept_candidate(
                    candidate, acquire=False, recovering=recovering
                )
                self.state = STATE_TRACK
                self.ever_locked = True
                self.acquire_count = TRACK_CFG["acquire_frames"]
                self.pending = None
                source = "DETECT_LOW" if low_used else "DETECT_HI"
                confidence = candidate["quality"]
        elif self.kalman.initialized() and self.state in (STATE_TRACK, STATE_COAST):
            self.detector_miss_count += 1
            self.strict_miss_count += 1
            self.miss_frames += 1
            velocity_decay = TRACK_CFG["coast_velocity_decay"]
            self.kalman.x.vel *= velocity_decay
            self.kalman.y.vel *= velocity_decay
            self.clamp_kalman_velocity()
            if self.miss_frames <= TRACK_CFG["max_coast_frames"]:
                self.state = STATE_COAST
                source = "PREDICT"
                confidence = max(
                    0.0,
                    1.0
                    - self.miss_frames
                    / float(TRACK_CFG["max_coast_frames"] + 1),
                )
            else:
                self.state = STATE_LOST
                self.hard_miss_count += 1
        else:
            if self.state == STATE_ACQUIRE:
                self.acquire_count = 0
                self.pending = None
                self.kalman.reset()
                self.last_box = None
                self.last_box_det = None
                self.last_area = None
                self.last_aspect = None
                self.raw_center = None
                self.filtered_center = None
                self.state = STATE_SEARCH
            elif self.state == STATE_LOST:
                self.miss_frames += 1

        if (
            self.kalman.initialized()
            and ticks_diff_ms(time.ticks_ms(), self.last_visual_ms)
            > TRACK_CFG["keep_track_ms"]
        ):
            self.kalman.reset()
            self.last_box = None
            self.last_box_det = None
            self.last_area = None
            self.last_aspect = None
            self.raw_center = None
            self.filtered_center = None
            self.state = STATE_SEARCH
            self.acquire_count = 0
            self.pending = None
            source = "NONE"

        center = None
        velocity = (0.0, 0.0)
        if self.kalman.initialized() and self.state != STATE_LOST:
            center = self.control_center()
            velocity = self.kalman.velocity()

        self.last_confidence = (
            float(confidence) if confidence > 0.0 else self.last_confidence
        )
        total_tracker_ms = perf_ms(step_start_us)
        return {
            "center": center,
            "raw_center": self.raw_center,
            "filtered_center": self.filtered_center,
            "velocity": velocity,
            "box": self.last_box,
            "box_det": self.last_box_det,
            "corners_det": self.last_corners_det,
            "candidate": candidate,
            "state": self.state,
            "source": source,
            "confidence": confidence,
            "candidates": candidates,
            "candidate_count": len(candidates),
            "miss_frames": self.miss_frames,
            "relaxed_used": low_used,
            "low_used": low_used,
            "strict_miss_count": self.strict_miss_count,
            "relaxed_recover_count": self.low_stage_count,
            "hard_miss_count": self.hard_miss_count,
            "relaxed_skipped_count": 0,
            "roi": roi,
            "roi_area": int(roi[2] * roi[3]),
            "full_scan": 1 if full_scan else 0,
            "detect_ms": detect_ms,
            "associate_ms": associate_ms,
            "refine_ms": refine_ms,
            "tracker_ms": total_tracker_ms,
            "corner_refine_fail_count": self.corner_refine_fail_count,
        }

# ============================================================
# 10. UART3 在线调参
# ============================================================
class UARTLink:
    def __init__(self):
        self.uart = None
        self.buffer = ""
        self.fpioa = None
        self.tx_packets = 0
        self.rx_packets = 0
        if not ENABLE_UART:
            print("External UART disabled")
            return

        try:
            self.fpioa = FPIOA()
            self.fpioa.set_function(UART_TX_PIN, FPIOA.UART3_TXD, ie=0, oe=1)
            self.fpioa.set_function(UART_RX_PIN, FPIOA.UART3_RXD, ie=1, oe=0)
            uart_id = getattr(UART, "UART3", UART_ID)
            try:
                self.uart = UART(
                    uart_id, baudrate=int(UART_BAUD), bits=8,
                    parity=None, stop=1, timeout=0
                )
            except TypeError:
                self.uart = UART(uart_id, baudrate=int(UART_BAUD))
            print("External UART ready: GPIO%d(TX) GPIO%d(RX) UART3 @ %d" % (
                UART_TX_PIN, UART_RX_PIN, UART_BAUD
            ))
        except Exception as e:
            print("UART3 init failed; tracking continues without tuning UART:", e)
            self.uart = None

    def is_ready(self):
        return self.uart is not None

    def send(self, text):
        if self.uart is None:
            return False
        try:
            self.uart.write((str(text) + "\r\n").encode("utf-8"))
            self.tx_packets += 1
            return True
        except Exception as e:
            print("UART write failed:", e)
            return False

    def send_display_help(self):
        if self.uart is None or not SEND_DISPLAY_HELP:
            return
        self.send("[plot-clear]")
        lines = (
            "K230 D36A tracker v3.4 anti-interference",
            "CH1 err_x CH2 err_y CH3 x_target_hz CH4 y_target_hz",
            "CH5 x_output_hz CH6 y_output_hz CH7 target_x CH8 target_y",
            "CH9 state:0 lost 1 acquire 2 track 3 coast 4 stop; CH10 fps",
            "PID: xKp xKi xKd yKp yKi yKd xDead yDead",
            "Motion: xMinHz yMinHz xMaxHz yMaxHz xAccel yAccel xDecel yDecel",
            "Limits: xLimitSteps yLimitSteps xLimitDeg yLimitDeg; [motor,get_limit]",
            "Wide test defaults: X ±360deg, Y ±160deg; STAT limit=(x,y) shows soft-limit hit",
            "Final v3.6 tuning: stronger X dynamic tracking; vision thresholds kept strict",
            "Adaptive: xNearCap yNearCap xMidCap yMidCap xFarCap yFarCap",
            "Brake: brakeTime predictStop approachMaxErr errorFloor xCoastMax yCoastMax",
            "Coast: coast1 coast2 coast3 coast4 coast5 coast6 coastPredict",
            "Edge: coastOutErr coastOutVel xCoastOutMax yCoastOutMax",
            "Recovery: coastRefGain coastRefMargin recoverVar lowVar xVelCap yVelCap; diagEnable",
            "Feedforward: xFF yFF lead speedScale; direction: xReverse yReverse",
            "Vision: rectSearchTh rectTrackTh rectCoastTh qHigh qLow minContrast",
            "ROI: roiScaleW roiScaleH roiMinW roiMinH fullScanN refineEvery perfEnable",
            "plotMode 0=control 1=Kalman/recovery 2=stepper diagnostic",
            "Commands: [key,start] [key,stop] [sv,estop] [sv,restart] [motor,test] [motor,jog,x,80,500]",
        )
        y = 8
        for line in lines:
            self.send("[display,8,%d,%s,14]" % (y, line))
            y += 18

    def read_packets(self, max_packets=UART_MAX_PACKETS_PER_LOOP):
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
            last_start = self.buffer.rfind("[")
            self.buffer = self.buffer[last_start:] if last_start >= 0 else ""
        while len(packets) < int(max_packets) and "[" in self.buffer and "]" in self.buffer:
            start = self.buffer.find("[")
            end = self.buffer.find("]", start)
            if end <= start:
                break
            content = self.buffer[start + 1:end]
            self.buffer = self.buffer[end + 1:]
            parts = [part.strip() for part in content.split(",")]
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


# ============================================================
# 11. 主控制器
# ============================================================
class K230RectangleStepperTrackerV31:
    def __init__(self):
        self.cx0 = CAMERA_WIDTH // 2
        self.cy0 = CAMERA_HEIGHT // 2

        self.common_enable = CommonEnable()
        self.x_axis = StepperAxis(
            "X", X_STEP_PIN, X_DIR_PIN, X_POSITIVE_DIR_LEVEL, X_REVERSE,
            MOTION_CFG["x_min_hz"], MOTION_CFG["x_max_hz"],
            MOTION_CFG["x_accel_hz_s"], MOTION_CFG["x_decel_hz_s"],
            X_SOFT_LIMIT_STEPS,
        )
        self.y_axis = StepperAxis(
            "Y", Y_STEP_PIN, Y_DIR_PIN, Y_POSITIVE_DIR_LEVEL, Y_REVERSE,
            MOTION_CFG["y_min_hz"], MOTION_CFG["y_max_hz"],
            MOTION_CFG["y_accel_hz_s"], MOTION_CFG["y_decel_hz_s"],
            Y_SOFT_LIMIT_STEPS,
        )
        self.common_enable.enable()
        print("D36A EN1/EN2 must be tied to board 5V; GPIO35 unused")

        self.x_pid = VelocityPID(
            PID_CFG["x_kp"], PID_CFG["x_ki"], PID_CFG["x_kd"],
            MOTION_CFG["x_max_hz"], PID_CFG["integral_limit"],
            PID_CFG["derivative_tau_s"],
        )
        self.y_pid = VelocityPID(
            PID_CFG["y_kp"], PID_CFG["y_ki"], PID_CFG["y_kd"],
            MOTION_CFG["y_max_hz"], PID_CFG["integral_limit"],
            PID_CFG["derivative_tau_s"],
        )

        self.tracker = RectangleTracker()
        self.uart = UARTLink()

        self.tracking_enabled = True
        self.estop = False
        self.last_err_x = None
        self.last_err_y = None
        self.last_x_target_hz = 0.0
        self.last_y_target_hz = 0.0
        self.last_track_x_hz = 0.0
        self.last_track_y_hz = 0.0

        now = time.ticks_ms()
        self.last_frame_ms = now
        self.last_plot_ms = now
        self.last_diag_ms = now
        self.last_status_ms = now
        self.last_dt_ms = 0
        self.fps = 0.0
        self.last_perf_ms = now
        self.perf_capture_ms = 0.0
        self.perf_control_ms = 0.0
        self.perf_display_ms = 0.0
        self.perf_total_ms = 0.0

    @staticmethod
    def crossed_zero(previous, current):
        if previous is None:
            return False
        return (previous > 0.0 and current < 0.0) or (previous < 0.0 and current > 0.0)

    def reset_pid_memory(self):
        self.x_pid.reset()
        self.y_pid.reset()
        self.last_err_x = None
        self.last_err_y = None

    def stop_motion(self, hard=True):
        self.last_x_target_hz = 0.0
        self.last_y_target_hz = 0.0
        self.last_track_x_hz = 0.0
        self.last_track_y_hz = 0.0
        if hard:
            self.x_axis.hard_stop()
            self.y_axis.hard_stop()
        else:
            self.x_axis.set_target_hz(0.0)
            self.y_axis.set_target_hz(0.0)
        self.reset_pid_memory()

    def stop_and_reset_tracker(self, zero_virtual=False):
        self.stop_motion(hard=True)
        self.tracker.reset()
        if zero_virtual:
            self.x_axis.zero_virtual_position()
            self.y_axis.zero_virtual_position()
        print("stepper stopped; tracker reset; zero_virtual=%s" % zero_virtual)

    def emergency_stop(self):
        self.estop = True
        self.stop_motion(hard=True)
        self.common_enable.disable()

    def restart_after_estop(self):
        self.common_enable.enable()
        self.estop = False
        self.stop_motion(hard=True)

    def adaptive_axis_cap(self, abs_error, near_cap, mid_cap, far_cap, max_hz):
        """按像素误差平滑计算当前允许的最大 STEP 频率。"""
        near_e = MOTION_CFG["near_error_px"]
        mid_e = MOTION_CFG["mid_error_px"]
        far_e = MOTION_CFG["far_error_px"]

        if abs_error <= near_e:
            cap = near_cap
        elif abs_error <= mid_e:
            ratio = (abs_error - near_e) / max(1.0, mid_e - near_e)
            cap = near_cap + ratio * (mid_cap - near_cap)
        elif abs_error <= far_e:
            ratio = (abs_error - mid_e) / max(1.0, far_e - mid_e)
            cap = mid_cap + ratio * (far_cap - mid_cap)
        else:
            cap = far_cap

        return clamp(cap, 0.0, max_hz)

    def compute_axis_target(self, pid, error_px, velocity_px_s, dt,
                            deadzone_px, ff_gain, ff_limit_hz,
                            min_hz, max_hz, previous_error,
                            near_cap_hz, mid_cap_hz, far_cap_hz):
        if abs(error_px) <= deadzone_px:
            pid.reset()
            return 0.0, False

        crossed = self.crossed_zero(previous_error, error_px)
        if crossed and abs(error_px) < ZERO_CROSS_BRAKE_PX:
            pid.reset()
            return 0.0, True

        pid.output_limit = float(max_hz)
        target = pid.update(error_px, dt)

        ff = 0.0
        if abs(velocity_px_s) > FAST_VEL_START_PX_S and abs(error_px) > FAST_ERR_START_PX:
            ff = clamp(ff_gain * velocity_px_s, -ff_limit_hz, ff_limit_hz)
            if target * ff < 0.0 and abs(error_px) < 80.0:
                ff *= 0.30
        target = (target + ff) * MOTION_CFG["speed_scale"]
        target = clamp(target, -max_hz, max_hz)

        abs_error = abs(error_px)
        adaptive_cap = self.adaptive_axis_cap(
            abs_error, near_cap_hz, mid_cap_hz, far_cap_hz, max_hz
        )
        target = clamp(target, -adaptive_cap, adaptive_cap)

        # 远离中心的误差不能因为PID/前馈抵消或错误速度预测而完全停住。
        # 这次数据中多次出现 |err|>100px 但 outHz=0，造成动态滞后。
        if abs_error >= MOTION_CFG["error_floor_start_px"] and abs(target) < min_hz:
            target = min_hz if error_px > 0.0 else -min_hz

        # error 与 velocity 异号表示目标正在向画面中心靠近。
        # 只在接近中心时提前制动；大误差区禁止“预计穿越中心→直接归零”。
        if (
            abs_error <= MOTION_CFG["approach_max_error_px"]
            and error_px * velocity_px_s < 0.0
            and abs(velocity_px_s) > 1.0
        ):
            predicted_error = error_px + velocity_px_s * MOTION_CFG["approach_predict_s"]
            if error_px * predicted_error <= 0.0:
                pid.reset()
                return 0.0, True
            time_to_center = abs(error_px) / max(1.0, abs(velocity_px_s))
            brake_time = MOTION_CFG["approach_brake_time_s"]
            if time_to_center < brake_time:
                scale = clamp(
                    time_to_center / max(0.01, brake_time),
                    MOTION_CFG["approach_min_scale"], 1.0,
                )
                target *= scale

        if 0.0 < abs(target) < min_hz:
            target = min_hz if target > 0.0 else -min_hz
        return target, False

    def update_tracking_control(self, result, dt):
        if self.estop or result["center"] is None:
            self.stop_motion(hard=True)
            return

        tx, ty = result["center"]
        err_x = float(tx - self.cx0)
        err_y = float(ty - self.cy0)
        vx, vy = result["velocity"]

        x_target, _ = self.compute_axis_target(
            self.x_pid, err_x, vx, dt,
            PID_CFG["x_deadzone_px"],
            MOTION_CFG["x_ff"], MOTION_CFG["x_ff_limit_hz"],
            self.x_axis.min_hz, self.x_axis.max_hz,
            self.last_err_x,
            MOTION_CFG["x_near_cap_hz"],
            MOTION_CFG["x_mid_cap_hz"],
            MOTION_CFG["x_far_cap_hz"],
        )
        y_target, _ = self.compute_axis_target(
            self.y_pid, err_y, vy, dt,
            PID_CFG["y_deadzone_px"],
            MOTION_CFG["y_ff"], MOTION_CFG["y_ff_limit_hz"],
            self.y_axis.min_hz, self.y_axis.max_hz,
            self.last_err_y,
            MOTION_CFG["y_near_cap_hz"],
            MOTION_CFG["y_mid_cap_hz"],
            MOTION_CFG["y_far_cap_hz"],
        )

        self.last_err_x = err_x
        self.last_err_y = err_y
        self.last_x_target_hz = self.x_axis.set_target_hz(x_target)
        self.last_y_target_hz = self.y_axis.set_target_hz(y_target)
        self.last_track_x_hz = self.last_x_target_hz
        self.last_track_y_hz = self.last_y_target_hz
        self.x_axis.update(dt)
        self.y_axis.update(dt)

    def coast_factor_for_frame(self, coast_frame, outward=False):
        """返回普通COAST或向外逃逸COAST的逐帧衰减系数。"""
        coast_frame = int(clamp(coast_frame, 1, 6))
        if outward:
            return MOTION_CFG["coast_out_scale_%d" % coast_frame]
        return MOTION_CFG["coast_scale_%d" % coast_frame]

    def compute_coast_axis_target(self, error_px, velocity_px_s, deadzone_px,
                                  kp, ff_gain, coast_frame,
                                  min_hz, normal_max_hz, outward_max_hz,
                                  edge_error_px, last_track_hz, axis_max_hz):
        """根据目标运动方向选择制动COAST或边缘恢复COAST。"""
        if abs(error_px) <= deadzone_px:
            return 0.0

        moving_outward = (
            error_px * velocity_px_s > 0.0
            and abs(error_px) >= MOTION_CFG["coast_outward_error_px"]
            and abs(velocity_px_s) >= MOTION_CFG["coast_outward_velocity_px_s"]
        )

        if moving_outward:
            factor = self.coast_factor_for_frame(coast_frame, outward=True)
            if factor <= 0.0:
                return 0.0

            target = (
                kp * error_px
                + MOTION_CFG["coast_out_ff_scale"] * ff_gain * velocity_px_s
            ) * factor

            reference_abs = abs(float(last_track_hz))
            if abs(error_px) >= edge_error_px:
                # 真正靠近画面边缘时允许继续追赶，但仍以最后有效TRACK速度为基准。
                reference_cap = (
                    reference_abs * MOTION_CFG["coast_reference_gain"]
                    + MOTION_CFG["coast_edge_reference_margin_hz"]
                )
                cap_hz = min(
                    outward_max_hz * MOTION_CFG["coast_edge_boost"],
                    max(normal_max_hz, reference_cap),
                    axis_max_hz,
                )
            else:
                # 非边缘漏检不得从低速/零速突然重新加速到数百Hz。
                reference_cap = (
                    reference_abs * MOTION_CFG["coast_reference_gain"]
                    + MOTION_CFG["coast_reference_margin_hz"]
                )
                cap_hz = min(
                    outward_max_hz,
                    max(min_hz, reference_cap),
                    axis_max_hz,
                )
            target = clamp(target, -cap_hz, cap_hz)
        else:
            factor = self.coast_factor_for_frame(coast_frame, outward=False)
            if factor <= 0.0:
                return 0.0

            # 正在向中心靠近且预计即将跨越中心时立即制动。
            predicted_error = (
                error_px + velocity_px_s * MOTION_CFG["coast_predict_stop_s"]
            )
            if error_px * predicted_error <= 0.0:
                return 0.0

            target = (
                kp * error_px
                + MOTION_CFG["coast_ff_scale"] * ff_gain * velocity_px_s
            ) * factor
            target = clamp(target, -normal_max_hz, normal_max_hz)

        if 0.0 < abs(target) < min_hz:
            target = min_hz if target > 0.0 else -min_hz
        return target

    def update_coast_control(self, result, dt):
        if (
            self.estop
            or not MOTION_CFG["coast_enable"]
            or result.get("center") is None
        ):
            self.stop_motion(hard=True)
            return

        coast_frame = int(result.get("miss_frames", 1))
        tx, ty = result["center"]
        vx, vy = result.get("velocity", (0.0, 0.0))
        err_x = float(tx - self.cx0)
        err_y = float(ty - self.cy0)

        x_target = self.compute_coast_axis_target(
            err_x, vx, PID_CFG["x_deadzone_px"],
            self.x_pid.kp, MOTION_CFG["x_ff"], coast_frame,
            self.x_axis.min_hz,
            MOTION_CFG["x_coast_max_hz"],
            MOTION_CFG["x_coast_out_max_hz"],
            MOTION_CFG["x_edge_error_px"],
            self.last_track_x_hz,
            self.x_axis.max_hz,
        )
        y_target = self.compute_coast_axis_target(
            err_y, vy, PID_CFG["y_deadzone_px"],
            self.y_pid.kp, MOTION_CFG["y_ff"], coast_frame,
            self.y_axis.min_hz,
            MOTION_CFG["y_coast_max_hz"],
            MOTION_CFG["y_coast_out_max_hz"],
            MOTION_CFG["y_edge_error_px"],
            self.last_track_y_hz,
            self.y_axis.max_hz,
        )

        self.last_x_target_hz = self.x_axis.set_target_hz(x_target)
        self.last_y_target_hz = self.y_axis.set_target_hz(y_target)
        self.x_axis.update(dt)
        self.y_axis.update(dt)

    def _jog_axis_blocking(self, axis, signed_hz, duration_ms):
        """低速短时点动。用于上电自检和串口手动测试。"""
        self.stop_motion(hard=True)
        axis.set_enabled(True)
        axis.set_target_hz(float(signed_hz))

        start_ms = time.ticks_ms()
        last_ms = start_ms
        while time.ticks_diff(time.ticks_ms(), start_ms) < int(duration_ms):
            now_ms = time.ticks_ms()
            dt = max(0.001, time.ticks_diff(now_ms, last_ms) / 1000.0)
            last_ms = now_ms
            axis.update(dt)
            time.sleep_ms(int(SELF_TEST_LOOP_DT_MS))

        axis.hard_stop()
        time.sleep_ms(int(SELF_TEST_PAUSE_MS))

    def startup_motor_self_test(self):
        """启动时自动让 X/Y 轴正反各点动一次；不依赖视觉目标。"""
        if not STARTUP_MOTOR_SELF_TEST:
            print("Motor self-test skipped")
            return

        print("--- D36A startup motor self-test begin ---")
        print("EN1/EN2 must already be tied to D36A 5V; motors should have holding torque")
        sequence = (
            ("X+", self.x_axis, +SELF_TEST_HZ),
            ("X-", self.x_axis, -SELF_TEST_HZ),
            ("Y+", self.y_axis, +SELF_TEST_HZ),
            ("Y-", self.y_axis, -SELF_TEST_HZ),
        )
        for label, axis, hz in sequence:
            print("SELFTEST %s %.0fHz %dms" % (label, hz, SELF_TEST_RUN_MS))
            self._jog_axis_blocking(axis, hz, SELF_TEST_RUN_MS)

        self.stop_motion(hard=True)
        self.x_axis.zero_virtual_position()
        self.y_axis.zero_virtual_position()
        print("--- D36A startup motor self-test end; virtual steps reset ---")

    def handle_uart(self):
        packets = self.uart.read_packets(UART_MAX_PACKETS_PER_LOOP)
        latest_slider = {}
        commands = []
        for parts in packets:
            if parts and parts[0].strip().lower() == "slider" and len(parts) >= 3:
                latest_slider[parts[1].strip().lower()] = parts[2]
            else:
                commands.append(parts)
        for name, value in latest_slider.items():
            self.apply_slider(name, value)

        for parts in commands:
            if not parts:
                continue
            typ = parts[0].strip().lower()
            cmd = parts[1].strip().lower() if len(parts) >= 2 else ""

            if typ in ("sv", "key", "stepper", "motor") and cmd in ("estop", "emergency"):
                self.emergency_stop()
                self.uart.send("[stepper,estop,1]")
            elif typ in ("sv", "stepper", "motor") and cmd in ("restart", "resume"):
                self.restart_after_estop()
                self.uart.send("[stepper,estop,0]")
            elif typ == "key" and cmd in ("start", "mode"):
                self.tracking_enabled = True
                self.restart_after_estop()
                self.uart.send("[stepper,tracking,1]")
            elif typ == "key" and cmd == "stop":
                self.tracking_enabled = False
                self.stop_motion(hard=True)
                self.uart.send("[stepper,tracking,0]")
            elif typ in ("stepper", "motor") and cmd == "zero":
                self.stop_motion(hard=True)
                self.x_axis.zero_virtual_position()
                self.y_axis.zero_virtual_position()
                self.uart.send("[stepper,zero,ok]")
            elif typ in ("stepper", "motor") and cmd in ("stop", "reset"):
                self.stop_and_reset_tracker(zero_virtual=False)
                self.uart.send("[stepper,stop,ok]")
            elif typ in ("stepper", "motor") and cmd in ("test", "selftest"):
                self.common_enable.enable()
                self.estop = False
                self.startup_motor_self_test()
                self.uart.send("[stepper,selftest,ok]")
            elif typ in ("stepper", "motor") and cmd == "jog" and len(parts) >= 5:
                axis_name = parts[2].strip().lower()
                try:
                    jog_hz = float(parts[3])
                    jog_ms = int(float(parts[4]))
                    jog_hz = clamp(jog_hz, -450.0, 450.0)
                    jog_ms = int(clamp(jog_ms, 50, 3000))
                    axis = self.x_axis if axis_name == "x" else self.y_axis
                    self.common_enable.enable()
                    self.estop = False
                    self._jog_axis_blocking(axis, jog_hz, jog_ms)
                    self.uart.send("[stepper,jog,%s,%.0f,%d,ok]" % (axis_name, jog_hz, jog_ms))
                except Exception as e:
                    self.uart.send("[stepper,jog,error,%s]" % str(e))
            elif typ == "servo" and cmd == "center":
                # 兼容旧网页命令，但无编码器时不能自动回中。
                self.stop_and_reset_tracker(zero_virtual=False)
                self.uart.send("[stepper,stopped,no_home_reference]")
            elif typ in ("stepper", "motor", "servo") and cmd in ("get_state", "get_angle"):
                self.uart.send(
                    "[stepper,state,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%d]" % (
                        self.x_axis.target_hz, self.y_axis.target_hz,
                        self.x_axis.applied_hz, self.y_axis.applied_hz,
                        self.x_axis.virtual_steps, self.y_axis.virtual_steps,
                        self.x_axis.limit_hit, self.y_axis.limit_hit,
                    )
                )
            elif typ in ("stepper", "motor", "servo") and cmd in ("get_limit", "get_limits"):
                x_limit_deg = (
                    self.x_axis.soft_limit_steps * 360.0 / float(PULSES_PER_REV)
                )
                y_limit_deg = (
                    self.y_axis.soft_limit_steps * 360.0 / float(PULSES_PER_REV)
                )
                self.uart.send(
                    "[stepper,limit,%.1f,%.1f,%.2f,%.2f]" % (
                        self.x_axis.soft_limit_steps,
                        self.y_axis.soft_limit_steps,
                        x_limit_deg,
                        y_limit_deg,
                    )
                )
            elif typ in ("stepper", "motor", "servo") and cmd == "get_pid":
                self.uart.send(
                    "[pid,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.1f,%.1f,%.1f,%.1f,%.2f]" % (
                        self.x_pid.kp, self.x_pid.ki, self.x_pid.kd,
                        self.y_pid.kp, self.y_pid.ki, self.y_pid.kd,
                        self.x_axis.min_hz, self.y_axis.min_hz,
                        self.x_axis.max_hz, self.y_axis.max_hz,
                        MOTION_CFG["speed_scale"],
                    )
                )
            elif typ == "system" and cmd == "ping":
                self.uart.send("[system,pong]")
            else:
                self.uart.send("[ack,error,unknown_command,%s,%s]" % (typ, cmd))

    def apply_slider(self, name, raw_value):
        global PLOT_INTERVAL_MS, DIAG_INTERVAL_MS, PLOT_MODE, ENABLE_DIAG_PACKET, ENABLE_PERF_PACKET
        try:
            value = float(raw_value)
        except Exception:
            self.uart.send("[ack,error,bad_value,%s]" % name)
            return

        ok = True
        reset_pid = False
        name = name.strip().lower()

        if name in ("xkp", "x_kp", "pankp", "pan_kp"):
            self.x_pid.kp = clamp(value, 0.0, 20.0); reset_pid = True
        elif name in ("xki", "x_ki", "panki", "pan_ki"):
            self.x_pid.ki = clamp(value, 0.0, 10.0); reset_pid = True
        elif name in ("xkd", "x_kd", "pankd", "pan_kd"):
            self.x_pid.kd = clamp(value, 0.0, 5.0); reset_pid = True
        elif name in ("ykp", "y_kp", "tiltkp", "tilt_kp"):
            self.y_pid.kp = clamp(value, 0.0, 20.0); reset_pid = True
        elif name in ("yki", "y_ki", "tiltki", "tilt_ki"):
            self.y_pid.ki = clamp(value, 0.0, 10.0); reset_pid = True
        elif name in ("ykd", "y_kd", "tiltkd", "tilt_kd"):
            self.y_pid.kd = clamp(value, 0.0, 5.0); reset_pid = True
        elif name == "kp":
            self.x_pid.kp = self.y_pid.kp = clamp(value, 0.0, 20.0); reset_pid = True
        elif name == "ki":
            self.x_pid.ki = self.y_pid.ki = clamp(value, 0.0, 10.0); reset_pid = True
        elif name == "kd":
            self.x_pid.kd = self.y_pid.kd = clamp(value, 0.0, 5.0); reset_pid = True
        elif name in ("deadzone", "dead", "deadzonepx"):
            PID_CFG["x_deadzone_px"] = PID_CFG["y_deadzone_px"] = clamp(value, 0.0, 80.0)
            reset_pid = True
        elif name in ("xdead", "x_dead", "pandead", "pan_dead"):
            PID_CFG["x_deadzone_px"] = clamp(value, 0.0, 80.0); reset_pid = True
        elif name in ("ydead", "y_dead", "tiltdead", "tilt_dead"):
            PID_CFG["y_deadzone_px"] = clamp(value, 0.0, 80.0); reset_pid = True
        elif name in ("xff", "x_ff", "panff"):
            MOTION_CFG["x_ff"] = clamp(value, 0.0, 3.0)
        elif name in ("yff", "y_ff", "tiltff"):
            MOTION_CFG["y_ff"] = clamp(value, 0.0, 3.0)
        elif name in ("lead", "predict", "leadtime"):
            TRACK_CFG["control_lead_s"] = clamp(value, 0.0, 0.20)
        elif name in ("speedscale", "speed_scale", "speed", "motorscale"):
            MOTION_CFG["speed_scale"] = clamp(value, 0.30, 2.50)
        elif name in ("xnearcap", "x_near_cap"):
            MOTION_CFG["x_near_cap_hz"] = clamp(value, 30.0, self.x_axis.max_hz)
        elif name in ("ynearcap", "y_near_cap"):
            MOTION_CFG["y_near_cap_hz"] = clamp(value, 30.0, self.y_axis.max_hz)
        elif name in ("xmidcap", "x_mid_cap"):
            MOTION_CFG["x_mid_cap_hz"] = clamp(value, 30.0, self.x_axis.max_hz)
        elif name in ("ymidcap", "y_mid_cap"):
            MOTION_CFG["y_mid_cap_hz"] = clamp(value, 30.0, self.y_axis.max_hz)
        elif name in ("xfarcap", "x_far_cap"):
            MOTION_CFG["x_far_cap_hz"] = clamp(value, 30.0, self.x_axis.max_hz)
        elif name in ("yfarcap", "y_far_cap"):
            MOTION_CFG["y_far_cap_hz"] = clamp(value, 30.0, self.y_axis.max_hz)
        elif name in ("braketime", "approach_brake_time"):
            MOTION_CFG["approach_brake_time_s"] = clamp(value, 0.03, 0.60)
        elif name in ("predictstop", "approach_predict"):
            MOTION_CFG["approach_predict_s"] = clamp(value, 0.0, 0.25)
        elif name in ("approachmaxerr", "approach_max_error"):
            MOTION_CFG["approach_max_error_px"] = clamp(value, 20.0, 180.0)
        elif name in ("errorfloor", "error_floor_start"):
            MOTION_CFG["error_floor_start_px"] = clamp(value, 6.0, 80.0)
        elif name in ("xcoastmax", "x_coast_max"):
            MOTION_CFG["x_coast_max_hz"] = clamp(value, 0.0, self.x_axis.max_hz)
        elif name in ("ycoastmax", "y_coast_max"):
            MOTION_CFG["y_coast_max_hz"] = clamp(value, 0.0, self.y_axis.max_hz)
        elif name in ("coast1", "coast_scale_1"):
            MOTION_CFG["coast_scale_1"] = clamp(value, 0.0, 1.20)
        elif name in ("coast2", "coast_scale_2"):
            MOTION_CFG["coast_scale_2"] = clamp(value, 0.0, 1.20)
        elif name in ("coast3", "coast_scale_3"):
            MOTION_CFG["coast_scale_3"] = clamp(value, 0.0, 1.20)
        elif name in ("coast4", "coast_scale_4"):
            MOTION_CFG["coast_scale_4"] = clamp(value, 0.0, 1.20)
        elif name in ("coast5", "coast_scale_5"):
            MOTION_CFG["coast_scale_5"] = clamp(value, 0.0, 1.20)
        elif name in ("coast6", "coast_scale_6"):
            MOTION_CFG["coast_scale_6"] = clamp(value, 0.0, 1.20)
        elif name in ("coastpredict", "coast_predict_stop"):
            MOTION_CFG["coast_predict_stop_s"] = clamp(value, 0.0, 0.25)
        elif name in ("coastouterr", "coast_outward_error"):
            MOTION_CFG["coast_outward_error_px"] = clamp(value, 5.0, 250.0)
        elif name in ("coastoutvel", "coast_outward_velocity"):
            MOTION_CFG["coast_outward_velocity_px_s"] = clamp(value, 0.0, 1000.0)
        elif name in ("xcoastoutmax", "x_coast_out_max"):
            MOTION_CFG["x_coast_out_max_hz"] = clamp(
                value, 0.0, self.x_axis.max_hz
            )
        elif name in ("ycoastoutmax", "y_coast_out_max"):
            MOTION_CFG["y_coast_out_max_hz"] = clamp(
                value, 0.0, self.y_axis.max_hz
            )
        elif name in ("coastedgeboost", "coast_edge_boost"):
            MOTION_CFG["coast_edge_boost"] = clamp(value, 1.0, 1.40)
        elif name in ("coastrefgain", "coast_reference_gain"):
            MOTION_CFG["coast_reference_gain"] = clamp(value, 0.80, 1.60)
        elif name in ("coastrefmargin", "coast_reference_margin"):
            MOTION_CFG["coast_reference_margin_hz"] = clamp(value, 0.0, 250.0)
        elif name in ("relaxedstride", "relaxed_scan_stride"):
            RECT_CFG["full_scan_after_miss"] = int(clamp(round(value), 1, 6))
        elif name in ("recovervar", "measurement_var_recover_scale"):
            TRACK_CFG["measurement_var_recover_scale"] = clamp(value, 1.0, 4.0)
        elif name in ("lowvar", "measurement_var_low_scale"):
            TRACK_CFG["measurement_var_low_scale"] = clamp(value, 1.0, 4.0)
        elif name in ("xvelcap", "max_velocity_x"):
            TRACK_CFG["max_velocity_x_px_s"] = clamp(value, 100.0, 1200.0)
        elif name in ("yvelcap", "max_velocity_y"):
            TRACK_CFG["max_velocity_y_px_s"] = clamp(value, 100.0, 1200.0)
        elif name in ("xminhz", "x_min_hz", "panminhz"):
            self.x_axis.min_hz = clamp(value, 5.0, self.x_axis.max_hz)
        elif name in ("yminhz", "y_min_hz", "tiltminhz"):
            self.y_axis.min_hz = clamp(value, 5.0, self.y_axis.max_hz)
        elif name in ("minhz", "min_hz"):
            v = clamp(value, 5.0, min(self.x_axis.max_hz, self.y_axis.max_hz))
            self.x_axis.min_hz = self.y_axis.min_hz = v
        elif name in ("xmaxhz", "x_max_hz", "panmaxhz"):
            self.x_axis.max_hz = clamp(value, self.x_axis.min_hz, 5000.0)
            self.x_pid.output_limit = self.x_axis.max_hz
        elif name in ("ymaxhz", "y_max_hz", "tiltmaxhz"):
            self.y_axis.max_hz = clamp(value, self.y_axis.min_hz, 5000.0)
            self.y_pid.output_limit = self.y_axis.max_hz
        elif name in ("maxhz", "max_hz", "limit", "outputlimit", "output_limit"):
            v = clamp(value, max(self.x_axis.min_hz, self.y_axis.min_hz), 5000.0)
            self.x_axis.max_hz = self.y_axis.max_hz = v
            self.x_pid.output_limit = self.y_pid.output_limit = v
        elif name in ("xaccel", "x_accel"):
            self.x_axis.accel_hz_s = clamp(value, 20.0, 30000.0)
        elif name in ("yaccel", "y_accel"):
            self.y_axis.accel_hz_s = clamp(value, 20.0, 30000.0)
        elif name in ("xdecel", "x_decel"):
            self.x_axis.decel_hz_s = clamp(value, 20.0, 30000.0)
        elif name in ("ydecel", "y_decel"):
            self.y_axis.decel_hz_s = clamp(value, 20.0, 30000.0)
        elif name in ("accel", "ramp"):
            v = clamp(value, 20.0, 30000.0)
            self.x_axis.accel_hz_s = self.y_axis.accel_hz_s = v
        elif name == "decel":
            v = clamp(value, 20.0, 30000.0)
            self.x_axis.decel_hz_s = self.y_axis.decel_hz_s = v
        elif name in ("xreverse", "x_reverse", "panreverse"):
            self.x_axis.reverse = bool(value >= 0.5)
            self.x_axis.hard_stop()
            self.x_axis.direction_sign = 0
        elif name in ("yreverse", "y_reverse", "tiltreverse"):
            self.y_axis.reverse = bool(value >= 0.5)
            self.y_axis.hard_stop()
            self.y_axis.direction_sign = 0
        elif name in ("xlimitsteps", "x_limit_steps"):
            self.x_axis.soft_limit_steps = max(0.0, value)
        elif name in ("ylimitsteps", "y_limit_steps"):
            self.y_axis.soft_limit_steps = max(0.0, value)
        elif name in ("xlimitdeg", "x_limit_deg", "panlimitdeg"):
            self.x_axis.soft_limit_steps = max(
                0.0, value * float(PULSES_PER_REV) / 360.0
            )
        elif name in ("ylimitdeg", "y_limit_deg", "tiltlimitdeg"):
            self.y_axis.soft_limit_steps = max(
                0.0, value * float(PULSES_PER_REV) / 360.0
            )
        elif name in ("coast", "coastframes", "maxlost"):
            TRACK_CFG["max_coast_frames"] = int(clamp(round(value), 0, 8))
        elif name in ("coasten", "coast_enable"):
            MOTION_CFG["coast_enable"] = bool(value >= 0.5)
        elif name in ("kalmanq", "accelnoise"):
            TRACK_CFG["kalman_accel_noise"] = clamp(value, 30.0, 2000.0)
        elif name in ("gate", "gatetracking", "trackgate"):
            TRACK_CFG["gate_tracking_px"] = clamp(value, 20.0, 400.0)
        elif name in ("reacquiregate", "gatereacquire"):
            TRACK_CFG["gate_reacquire_px"] = clamp(value, 40.0, 500.0)
        elif name in ("minarea", "area"):
            RECT_CFG["min_area_detect"] = clamp(value, 30.0, 12000.0)
        elif name in ("amin", "aspectmin", "ratiomin"):
            RECT_CFG["aspect_min"] = clamp(value, 1.0, 2.8)
        elif name in ("amax", "aspectmax", "ratiomax"):
            RECT_CFG["aspect_max"] = clamp(value, 1.05, 3.5)
        elif name in ("targetaspect", "targetratio", "aspect"):
            RECT_CFG["target_aspect"] = clamp(value, 1.0, 3.0)
        elif name in ("rectsearchth", "searchth", "rect_threshold_search"):
            RECT_CFG["threshold_search"] = int(clamp(round(value), 500, 50000))
        elif name in ("recttrackth", "trackth", "rect_threshold_track"):
            RECT_CFG["threshold_track"] = int(clamp(round(value), 500, 50000))
        elif name in ("rectcoastth", "coastth", "rect_threshold_coast"):
            RECT_CFG["threshold_coast"] = int(clamp(round(value), 500, 50000))
        elif name in ("qhigh", "qualityhigh"):
            RECT_CFG["quality_high"] = clamp(value, 0.10, 0.95)
        elif name in ("qlow", "qualitylow"):
            RECT_CFG["quality_low"] = clamp(value, 0.05, RECT_CFG["quality_high"])
        elif name in ("qsearch", "qualitysearch", "searchquality"):
            RECT_CFG["quality_search_recover"] = clamp(value, 0.05, 0.95)
        elif name in ("lowring", "low_stage_min_ring"):
            RECT_CFG["low_stage_min_ring"] = clamp(value, 0.0, 1.0)
        elif name in ("jumpguard", "jump_guard"):
            TRACK_CFG["jump_guard_px"] = clamp(value, 20.0, 260.0)
        elif name in ("cvangle", "cv_max_angle_cos"):
            RECT_CFG["cv_max_angle_cos"] = clamp(value, 0.15, 0.70)
        elif name in ("nativeen", "native_enable"):
            RECT_CFG["native_enable"] = bool(value >= 0.5)
            self.tracker.native_runtime_enabled = RECT_CFG["native_enable"]
            self.tracker.native_slow_count = 0
        elif name in ("mincontrast", "contrast"):
            RECT_CFG["min_contrast"] = clamp(value, -40.0, 120.0)
        elif name in ("roiscalew", "roi_scale_w"):
            RECT_CFG["roi_scale_w"] = clamp(value, 1.2, 6.0)
        elif name in ("roiscaleh", "roi_scale_h"):
            RECT_CFG["roi_scale_h"] = clamp(value, 1.2, 6.0)
        elif name in ("roiminw", "roi_min_w"):
            RECT_CFG["roi_min_w"] = int(clamp(round(value), 40, DETECT_WIDTH))
        elif name in ("roiminh", "roi_min_h"):
            RECT_CFG["roi_min_h"] = int(clamp(round(value), 40, DETECT_HEIGHT))
        elif name in ("fullscann", "full_scan_interval"):
            RECT_CFG["full_scan_interval"] = int(clamp(round(value), 1, 120))
        elif name in ("refineevery", "corner_refine_every"):
            RECT_CFG["corner_refine_every"] = int(clamp(round(value), 1, 20))
        elif name in ("refineenable", "corner_refine_enable"):
            RECT_CFG["corner_refine_enable"] = bool(value >= 0.5)
        elif name in ("perfen", "perfenable", "perf_enable"):
            ENABLE_PERF_PACKET = bool(value >= 0.5)
        elif name in ("plotmode", "plot_mode"):
            PLOT_MODE = int(clamp(round(value), 0, 2))
            self.uart.send("[plot-clear]")
        elif name in ("plotms", "plotinterval", "plot_interval"):
            PLOT_INTERVAL_MS = int(clamp(round(value), 20, 500))
        elif name in ("diagms", "diaginterval"):
            DIAG_INTERVAL_MS = int(clamp(round(value), 100, 2000))
        elif name in ("diagen", "diagenable", "diag_enable"):
            ENABLE_DIAG_PACKET = bool(value >= 0.5)
        else:
            ok = False

        if ok:
            if reset_pid:
                self.reset_pid_memory()
            self.uart.send("[ack,slider,%s,%s]" % (name, str(value)))
        else:
            self.uart.send("[ack,error,unknown_slider,%s]" % name)

    def update_time(self):
        now = time.ticks_ms()
        dt_ms = max(1, ticks_diff_ms(now, self.last_frame_ms))
        self.last_frame_ms = now
        self.last_dt_ms = dt_ms
        dt = clamp(dt_ms / 1000.0, CONTROL_DT_MIN, CONTROL_DT_MAX)
        inst_fps = 1000.0 / dt_ms
        self.fps = inst_fps if self.fps <= 0.0 else 0.85 * self.fps + 0.15 * inst_fps
        return now, dt

    def state_code(self, result):
        if self.estop or not self.tracking_enabled:
            return 4
        if result["state"] == STATE_ACQUIRE:
            return 1
        if result["state"] == STATE_TRACK:
            return 2
        if result["state"] == STATE_COAST:
            return 3
        return 0

    def send_plot(self, result):
        now = time.ticks_ms()
        if ticks_diff_ms(now, self.last_plot_ms) < PLOT_INTERVAL_MS:
            return
        self.last_plot_ms = now
        state_code = self.state_code(result)

        if PLOT_MODE == 1:
            raw = result.get("raw_center")
            filt = result.get("filtered_center")
            vx, vy = result.get("velocity", (0.0, 0.0))
            if raw is None or filt is None:
                lag_x, lag_y = 0.0, 0.0
            else:
                lag_x = float(raw[0] - filt[0])
                lag_y = float(raw[1] - filt[1])
            self.uart.send(
                "[plot,%d,%.2f,%.2f,%.1f,%.1f,%d,%d,%.3f,%d,%.2f]" % (
                    int(self.last_dt_ms), lag_x, lag_y, vx, vy,
                    int(result.get("miss_frames", 0)),
                    int(result.get("relaxed_used", 0)),
                    float(result.get("confidence", 0.0)),
                    state_code, self.fps,
                )
            )
            return

        if PLOT_MODE == 2:
            self.uart.send(
                "[plot,%.1f,%.1f,%.1f,%.1f,%d,%d,%.1f,%.1f,%d,%.2f]" % (
                    self.x_axis.target_hz, self.y_axis.target_hz,
                    self.x_axis.applied_hz, self.y_axis.applied_hz,
                    self.x_axis.direction_sign, self.y_axis.direction_sign,
                    self.x_axis.virtual_steps, self.y_axis.virtual_steps,
                    state_code, self.fps,
                )
            )
            return

        if result["center"] is None:
            tx, ty = 0.0, 0.0
            err_x, err_y = 0.0, 0.0
        else:
            tx, ty = result["center"]
            err_x = float(tx - self.cx0)
            err_y = float(ty - self.cy0)
        self.uart.send(
            "[plot,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%.2f]" % (
                err_x, err_y,
                self.x_axis.target_hz, self.y_axis.target_hz,
                self.x_axis.applied_hz, self.y_axis.applied_hz,
                tx, ty, state_code, self.fps,
            )
        )

    def send_diag(self, result):
        if not ENABLE_DIAG_PACKET or not self.uart.is_ready():
            return
        now = time.ticks_ms()
        if ticks_diff_ms(now, self.last_diag_ms) < DIAG_INTERVAL_MS:
            return
        self.last_diag_ms = now
        raw = result.get("raw_center")
        filt = result.get("filtered_center")
        vx, vy = result.get("velocity", (0.0, 0.0))
        raw_x, raw_y = (-1.0, -1.0) if raw is None else raw
        filt_x, filt_y = (-1.0, -1.0) if filt is None else filt
        self.uart.send(
            "[diag,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.3f,"
            "%.1f,%.1f,%.1f,%.1f,%d,%d,%.1f,%.1f,%d,%d,%d,%d,%d,%d,%d]" % (
                now, self.last_dt_ms,
                raw_x, raw_y, filt_x, filt_y, vx, vy,
                float(result.get("confidence", 0.0)),
                self.x_axis.target_hz, self.y_axis.target_hz,
                self.x_axis.applied_hz, self.y_axis.applied_hz,
                self.x_axis.direction_sign, self.y_axis.direction_sign,
                self.x_axis.virtual_steps, self.y_axis.virtual_steps,
                self.x_axis.limit_hit, self.y_axis.limit_hit,
                int(result.get("miss_frames", 0)),
                int(result.get("relaxed_used", 0)),
                int(result.get("strict_miss_count", 0)),
                int(result.get("relaxed_recover_count", 0)),
                int(result.get("hard_miss_count", 0)),
            )
        )

    def send_perf(self, result):
        if not ENABLE_PERF_PACKET or not self.uart.is_ready():
            return
        now = time.ticks_ms()
        if ticks_diff_ms(now, self.last_perf_ms) < PERF_INTERVAL_MS:
            return
        self.last_perf_ms = now
        self.uart.send(
            "[perf,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d]" % (
                self.perf_capture_ms,
                float(result.get("detect_ms", 0.0)),
                float(result.get("associate_ms", 0.0)),
                float(result.get("refine_ms", 0.0)),
                self.perf_control_ms,
                self.perf_display_ms,
                self.perf_total_ms,
                int(result.get("roi_area", 0)),
                int(result.get("full_scan", 0)),
                int(result.get("candidate_count", 0)),
                self.state_code(result),
            )
        )

    def draw(self, img, result):
        cx_det = int(round(logical_to_detect_x(self.cx0)))
        cy_det = int(round(logical_to_detect_y(self.cy0)))
        img.draw_line(cx_det - 10, cy_det, cx_det + 10, cy_det,
                      color=(0, 80, 255), thickness=1)
        img.draw_line(cx_det, cy_det - 10, cx_det, cy_det + 10,
                      color=(0, 80, 255), thickness=1)

        roi = result.get("roi")
        if roi is not None and not result.get("full_scan", 0):
            img.draw_rectangle(
                int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]),
                color=(80, 80, 255), thickness=1,
            )

        corners_det = result.get("corners_det")
        color = (
            (0, 255, 0)
            if result["state"] == STATE_TRACK
            else (255, 180, 0)
        )
        if corners_det is not None:
            pts = [(int(round(p[0])), int(round(p[1]))) for p in order_corners(corners_det)]
            for i in range(4):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % 4]
                img.draw_line(x0, y0, x1, y1, color=color, thickness=2)
            for p in pts:
                img.draw_circle(p[0], p[1], 2, color=(255, 255, 0), thickness=1)
        else:
            box_det = result.get("box_det")
            if box_det is not None:
                x, y, w, h = [int(v) for v in box_det]
                img.draw_rectangle(x, y, w, h, color=color, thickness=2)

        if result["center"] is not None:
            tx, ty = result["center"]
            tx_det = int(round(logical_to_detect_x(tx)))
            ty_det = int(round(logical_to_detect_y(ty)))
            img.draw_cross(
                tx_det, ty_det, color=(255, 0, 0), size=5, thickness=1
            )
            img.draw_line(
                cx_det, cy_det, tx_det, ty_det,
                color=(255, 80, 80), thickness=1,
            )

        img.draw_string_advanced(
            4, 4, 14,
            "%s/%s %.1ffps" % (
                STATE_NAME[result["state"]], result["source"], self.fps
            ),
            color=(255, 255, 255),
        )
        img.draw_string_advanced(
            4, 21, 13,
            "ROI=%d%% det=%.1fms cand=%d" % (
                int(100.0 * result.get("roi_area", 0)
                    / float(DETECT_WIDTH * DETECT_HEIGHT)),
                float(result.get("detect_ms", 0.0)),
                int(result.get("candidate_count", 0)),
            ),
            color=(255, 255, 255),
        )
        img.draw_string_advanced(
            4, 38, 13,
            "X %.0f/%.0f Y %.0f/%.0fHz" % (
                self.x_axis.target_hz, self.x_axis.applied_hz,
                self.y_axis.target_hz, self.y_axis.applied_hz,
            ),
            color=(255, 255, 255),
        )
        if self.estop:
            img.draw_string_advanced(4, 56, 18, "ESTOP", color=(255, 0, 0))
        elif not self.tracking_enabled:
            img.draw_string_advanced(
                4, 56, 16, "TRACKING STOPPED", color=(255, 180, 0)
            )

    def print_status(self, result):
        now = time.ticks_ms()
        if ticks_diff_ms(now, self.last_status_ms) < STATUS_INTERVAL_MS:
            return
        self.last_status_ms = now
        if result["center"] is None:
            print(
                "STAT state=%s found=0 cand=%d det=%.1fms roi=%d%% full=%d "
                "hz=(%.0f,%.0f) steps=(%.0f,%.0f) limit=(%d,%d) fps=%.1f" % (
                    STATE_NAME[result["state"]],
                    int(result.get("candidate_count", 0)),
                    float(result.get("detect_ms", 0.0)),
                    int(
                        100.0 * result.get("roi_area", 0)
                        / float(DETECT_WIDTH * DETECT_HEIGHT)
                    ),
                    int(result.get("full_scan", 0)),
                    self.x_axis.applied_hz, self.y_axis.applied_hz,
                    self.x_axis.virtual_steps, self.y_axis.virtual_steps,
                    self.x_axis.limit_hit, self.y_axis.limit_hit,
                    self.fps,
                )
            )
        else:
            tx, ty = result["center"]
            vx, vy = result["velocity"]
            print(
                "STAT state=%s src=%s err=(%d,%d) vel=(%.0f,%.0f) "
                "targetHz=(%.0f,%.0f) outHz=(%.0f,%.0f) steps=(%.0f,%.0f) limit=(%d,%d) fps=%.1f" % (
                    STATE_NAME[result["state"]], result["source"],
                    tx - self.cx0, ty - self.cy0, vx, vy,
                    self.x_axis.target_hz, self.y_axis.target_hz,
                    self.x_axis.applied_hz, self.y_axis.applied_hz,
                    self.x_axis.virtual_steps, self.y_axis.virtual_steps,
                    self.x_axis.limit_hit, self.y_axis.limit_hit,
                    self.fps,
                )
            )

    def run(self):
        sensor = None
        display_ok = False
        try:
            print("=" * 76)
            print("K230 D36A RECTANGLE STEPPER TRACKER V3.6 FINAL-RECT")
            print("Vision: 320x240 RGB888; UART/PID logical coordinates: 640x480")
            print("Detector: cv_lite ROI + stricter two-stage association + corner polygon draw")
            print("EN1/EN2 tied to D36A board 5V; GPIO35 unused")
            print("=" * 76)
            self.startup_motor_self_test()

            try:
                sensor = Sensor(
                    id=SENSOR_ID,
                    width=SENSOR_INPUT_WIDTH,
                    height=SENSOR_INPUT_HEIGHT,
                    fps=SENSOR_FPS,
                )
            except Exception:
                sensor = Sensor()

            sensor.reset()
            sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT)
            sensor.set_pixformat(Sensor.RGB888)

            if ENABLE_DISPLAY:
                # ST7701不支持320x240。优先使用IDE虚拟显示；
                # 若固件不支持VIRT，再退回ST7701固定800x480模式。
                try:
                    Display.init(
                        Display.VIRT,
                        width=DETECT_WIDTH,
                        height=DETECT_HEIGHT,
                        fps=60,
                        to_ide=True,
                    )
                    display_ok = True
                    print("Display ready: VIRT %dx%d to IDE" % (
                        DETECT_WIDTH, DETECT_HEIGHT
                    ))
                except Exception as virt_e:
                    try:
                        Display.init(
                            Display.ST7701,
                            width=800,
                            height=480,
                            to_ide=True,
                            quality=DISPLAY_QUALITY,
                        )
                        display_ok = True
                        print("Display ready: ST7701 800x480")
                    except Exception as lcd_e:
                        print(
                            "Display init failed: VIRT=%s; ST7701=%s"
                            % (virt_e, lcd_e)
                        )
                        display_ok = False

            MediaManager.init()
            sensor.run()
            time.sleep_ms(500)

            print("=" * 76)
            print("K230 D36A rectangle tracker v3.6 FINAL-RECT started")
            print("Vision RGB888 detect=%dx%d logical=%dx%d refine=%s" % (
                DETECT_WIDTH, DETECT_HEIGHT,
                CAMERA_WIDTH, CAMERA_HEIGHT,
                str(CV2_REFINE_AVAILABLE and RECT_CFG["corner_refine_enable"]),
            ))
            print("X: STEP GPIO%d DIR GPIO%d; Y: STEP GPIO%d DIR GPIO%d" % (
                X_STEP_PIN, X_DIR_PIN, Y_STEP_PIN, Y_DIR_PIN
            ))
            print("UART: GPIO32 TX / GPIO33 RX, 115200 8N1")
            print("EN1/EN2: tied to D36A board 5V; GPIO35 unused")
            print("Plot packet remains 10 channels in 640x480 logical coordinates")
            print("Perf packet: capture/detect/associate/refine/control/display/total")
            print("=" * 76)

            if self.uart.is_ready():
                self.uart.send_display_help()
                self.uart.send("[system,ready,k230_d36a_stepper_v3_1_compat_hybrid]")

            frame_count = 0
            while True:
                os.exitpoint()
                loop_start_us = time.ticks_us()
                now, dt = self.update_time()

                capture_start_us = time.ticks_us()
                img = sensor.snapshot()
                self.perf_capture_ms = perf_ms(capture_start_us)

                img_np = img.to_numpy_ref()
                result = self.tracker.step(img, img_np, dt)

                control_start_us = time.ticks_us()
                if self.tracking_enabled and not self.estop:
                    if (
                        result["state"] == STATE_TRACK
                        and result["center"] is not None
                    ):
                        self.update_tracking_control(result, dt)
                    elif (
                        result["state"] == STATE_COAST
                        and result["center"] is not None
                    ):
                        self.update_coast_control(result, dt)
                    else:
                        self.stop_motion(hard=True)
                else:
                    self.stop_motion(hard=True)
                self.perf_control_ms = perf_ms(control_start_us)

                self.handle_uart()
                self.send_plot(result)
                self.send_diag(result)
                self.print_status(result)

                self.perf_display_ms = 0.0
                frame_count += 1
                if (
                    display_ok
                    and frame_count % max(1, DISPLAY_EVERY_N_FRAMES) == 0
                ):
                    display_start_us = time.ticks_us()
                    self.draw(img, result)
                    Display.show_image(img)
                    self.perf_display_ms = perf_ms(display_start_us)

                self.perf_total_ms = perf_ms(loop_start_us)
                self.send_perf(result)

                if frame_count % GC_CHECK_INTERVAL_FRAMES == 0:
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
            self.emergency_stop()
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
            self.common_enable.deinit()
            print(
                "program exited; STEP PWM disabled; "
                "D36A remains hardware-enabled by EN=5V"
            )


if __name__ == "__main__":
    K230RectangleStepperTrackerV31().run()
