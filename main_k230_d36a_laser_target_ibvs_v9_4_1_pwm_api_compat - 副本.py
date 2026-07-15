# -*- coding: utf-8 -*-
"""
K230 + D36A 二维图像雅可比激光打靶 v9.4.1

运行环境：CanMV IDE K230 / MicroPython
视觉基准：main_k230_rect_track_uart_v13_speed_recovery.py
执行器：D36A 双路步进驱动板 + 两相步进电机

架构：同帧视觉端点闭环 + 2x2图像雅可比自动标定/在线更新 +
粗瞄连续速度控制 + 精瞄有限脉冲迭代 + 严格安全监督。
主控制与标定始终使用真实256x192检测坐标，不用放大的640x480伪精度。

固定引脚：
    GPIO42 -> D36A STEP1（X轴）
    GPIO26 -> D36A DIR1 （X轴）
    GPIO43 -> D36A STEP2（Y轴）
    GPIO34 -> D36A DIR2 （Y轴）
    GPIO32 -> USB-TTL RX（K230 UART3 TX）
    GPIO33 <- USB-TTL TX（K230 UART3 RX）
    GND    -> D36A GND、USB-TTL GND

D36A 的 EN1、EN2 不接 K230，必须并接到驱动板自身的 5V，保持硬件持续使能。
GPIO35用于激光TTL控制。严禁把 D36A 的 5V 接到任何 K230 GPIO。

主数据包（10通道，坐标均为256x192检测坐标）：
[plot,ex,ey,error_norm,x_cmd,y_cmd,laser_x,laser_y,target_x,target_y,state]
诊断包：[vision,...] [jacobian,...] [control,...] [perf,...] [fault,...]

状态：0 INIT；1 SEARCH_TARGET；2 LOCK_TARGET；3 SEARCH_LASER；
4 VERIFY_LASER；5 CALIBRATE_X；6 CALIBRATE_Y；7 COARSE_AIM；
8 FINE_AIM；9 LOCKED；10 REACQUIRE；11 FAULT；12 ESTOP。

UART命令：
[tracking,start/stop] [laser,on/off] [calibrate,start/reset]
[jacobian,print] [estop] [restart] [motor,test] [motor,jog,x|y,hz,ms]
[slider,coarseGain|fineGain|fineEnterPx|fineExitPx|lockXPx|lockYPx,value]
[slider,xMaxHz|yMaxHz|xAccel|yAccel|calXPulses|calYPulses,value]
[slider,fineMaxPulsesX|fineMaxPulsesY|jacobianBeta,value]

重要限制：
- 本系统没有编码器，virtual_steps 只是按输出频率积分得到的软件估计；
- 断电、堵转或失步后，virtual_steps 与真实位置可能不一致；
- 本版不执行自动回中；启动前应把云台手动放在安全中间位置；
- 上电手动置中后，以软件累计步数提供正负90度二级安全保护；它不是编码器。
- 启动、FAULT、ESTOP、异常和程序退出均关闭激光并停止双轴STEP；
- 未同时确认真实矩形和真实激光时禁止运动，精瞄动作最多100脉冲。
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
# 视觉在256x192上运行以降低延迟；PID、UART和历史数据仍使用640x480逻辑坐标。
# 所有检测尺度参数均由320x240基准自动缩放，避免降分辨率后门限失真。
DETECT_BASE_WIDTH = 320
DETECT_BASE_HEIGHT = 240
DETECT_WIDTH = 256
DETECT_HEIGHT = 192
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DETECT_SCALE_X = DETECT_WIDTH / float(DETECT_BASE_WIDTH)
DETECT_SCALE_Y = DETECT_HEIGHT / float(DETECT_BASE_HEIGHT)
DETECT_SCALE = min(DETECT_SCALE_X, DETECT_SCALE_Y)
DETECT_AREA_SCALE = DETECT_SCALE_X * DETECT_SCALE_Y
DETECT_TO_LOGICAL_X = CAMERA_WIDTH / float(DETECT_WIDTH)
DETECT_TO_LOGICAL_Y = CAMERA_HEIGHT / float(DETECT_HEIGHT)
IMAGE_SHAPE = [DETECT_HEIGHT, DETECT_WIDTH]

ENABLE_DISPLAY = False       # 闭环性能模式：关闭LCD绘制与传输，不影响UART曲线。
DISPLAY_TO_IDE = False       # IDE镜像需要JPEG编码，会明显占用主循环。
DISPLAY_QUALITY = 35
DISPLAY_EVERY_N_FRAMES = 5   # 显示只用于观察，控制和检测仍每帧执行。

ENABLE_UART = True
UART_BAUD = 115200
UART_TX_PIN = 32
UART_RX_PIN = 33
UART_ID = 3
PLOT_INTERVAL_MS = 50
PLOT_MODE = 0              # 0=控制；1=视觉/Kalman；2=步进驱动诊断
DIAG_INTERVAL_MS = 200
ENABLE_DIAG_PACKET = False
PERF_INTERVAL_MS = 500
ENABLE_PERF_PACKET = True
UART_MAX_PACKETS_PER_LOOP = 6
STATUS_INTERVAL_MS = 2000
SEND_DISPLAY_HELP = True
GC_CHECK_INTERVAL_FRAMES = 90
GC_FREE_THRESHOLD = 190000

SENSOR_ID = 2
SENSOR_INPUT_WIDTH = 1280
SENSOR_INPUT_HEIGHT = 960
SENSOR_FPS = 90

# v4.1关键修正：
# 矩形检测使用256x192 RGB888，供cv_lite和to_numpy_ref使用；
# 激光检测单独使用256x192 RGB565，因为当前CanMV v1.4.3的
# find_blobs颜色/LAB链路在RGB888通道上可能持续返回0候选。
RECT_CHANNEL = CAM_CHN_ID_0
LASER_CHANNEL = CAM_CHN_ID_1

# ============================================================
# 2. D36A STEP/DIR 硬件配置
# ============================================================
X_STEP_PIN = 42            # PWM0
Y_STEP_PIN = 43            # PWM1
X_PWM_CHANNEL = 0
Y_PWM_CHANNEL = 1
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
# K230 上频繁调用 PWM.freq() 可能重新装载硬件计数器。按自适应门限更新频率，
# 让脉冲列连续；低速至少2Hz变化才立即更新，细小变化最多70ms后同步。
PWM_FREQ_REL_DELTA = 0.015
PWM_FREQ_MAX_HOLD_MS = 70

# WHEELTEC 例程基于 1.8° 电机、16 细分：200*16=3200 pulse/rev。
MOTOR_FULL_STEPS_PER_REV = 200
MICROSTEP = 16
PULSES_PER_REV = MOTOR_FULL_STEPS_PER_REV * MICROSTEP

# 用户要求完全放开软件转角范围。总开关关闭后，串口限位滑块也不能重新启用。
SOFT_LIMITS_ENABLED = True
SOFT_LIMIT_DEG = 360.0
X_SOFT_LIMIT_STEPS = PULSES_PER_REV * SOFT_LIMIT_DEG / 360.0
Y_SOFT_LIMIT_STEPS = PULSES_PER_REV * SOFT_LIMIT_DEG / 360.0

# ============================================================
# 3. 矩形检测与目标估计参数（沿用 v13 基线）
# ============================================================
RECT_CFG = {
    # CanMV原生find_rects阈值；值越高越严格。
    "threshold_search": 2800,
    "threshold_track": 2100,
    "threshold_coast": 1600,
    "threshold_periodic_full": 2400,

    # 检测尺度下的基础几何门限。
    "min_area_detect": 120.0 * DETECT_AREA_SCALE,
    "max_area_ratio": 0.46,
    "min_w_detect": max(8, int(round(12 * DETECT_SCALE_X))),
    "min_h_detect": max(6, int(round(9 * DETECT_SCALE_Y))),
    "aspect_min": 1.05,
    "aspect_max": 2.38,
    "target_aspect": 1.50,

    # 候选质量分层：同一遍检测结果先关联高置信度，再关联低置信度。
    "quality_high": 0.38,
    "quality_low": 0.08,
    "quality_search_recover": 0.32,
    "max_candidates": 10,

    # cv_lite轮廓检测参数。原生find_rects在本机超过45ms时会自动停用。
    "cv_canny_low": 22,
    "cv_canny_high": 84,
    "cv_approx_epsilon": 0.0230,
    "cv_area_min_ratio": 0.0010,
    "cv_max_angle_cos": 0.46,
    "cv_gaussian_blur_size": 3,
    "cv_true_roi_enable": True,
    "cv_true_roi_max_area_ratio": 0.90,
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
    "roi_scale_w": 3.20,
    "roi_scale_h": 3.10,
    "roi_min_w": int(round(150 * DETECT_SCALE_X)),
    "roi_min_h": int(round(120 * DETECT_SCALE_Y)),
    "roi_velocity_lead_s": 0.160,
    "roi_velocity_margin_gain": 0.10,
    "roi_extra_margin": int(round(18 * DETECT_SCALE)),
    # 静止靶锁定后优先使用真实ROI；连续3帧漏检才回全画面，避免单帧抖动
    # 触发高开销全扫。周期全扫只用于发现明显场景变化。
    "full_scan_interval": 36,
    "full_scan_after_miss": 3,

    # 亚像素角点仅对最终选中候选运行，失败会自动退回原生角点。
    # 实机上锁定后开启cornerSubPix会使FPS由62持续降到2左右。
    # 使用原生角点 + Kalman估计，不在实时主链执行该高开销路径。
    "corner_refine_enable": False,
    "corner_refine_every": 4,
    "corner_refine_window": max(3, int(round(4 * DETECT_SCALE))),
    "corner_refine_max_shift": 7.0 * DETECT_SCALE,
    "corner_refine_pad": max(5, int(round(8 * DETECT_SCALE))),
}


TRACK_CFG = {
    # 固定跑道上的静止靶：高质量首帧直接建立控制，不再为运动目标等待多帧。
    "acquire_frames": 1,
    "reacquire_frames": 1,
    # 目标身份记忆与电机漏帧动作分离：矩形轨迹可保留较久，但电机仅短时桥接。
    "max_coast_frames": 12,
    "keep_track_ms": 5000,
    # 检测仍使用Kalman稳中心，但控制中心不做运动目标前瞻。
    "control_lead_s": 0.0,

    "gate_tracking_px": 150.0,
    "gate_reacquire_px": 300.0,
    "acquire_jump_px": 120.0,
    "max_area_ratio_tracking": 2.5,
    "max_area_ratio_reacquire": 5.4,
    "max_aspect_delta_tracking": 0.32,
    "max_aspect_delta_reacquire": 0.58,

    # 抗干扰跳变保护：非高质量、非黑白环结构的候选不得从上一帧目标处大幅跳转。
    "jump_guard_px": 95.0,
    "jump_guard_quality": 0.42,
    "jump_guard_ring": 0.18,
    "jump_guard_contrast": 9.0,
    "temporal_override_px": 65.0,

    "kalman_accel_noise": 420.0,
    "measurement_var_detect": 16.0,

    # COAST恢复帧与低置信度候选不应立即强校正，避免速度估计尖峰。
    "measurement_var_recover_scale": 1.20,
    "measurement_var_low_scale": 2.20,

    # 限制不可信的Kalman速度，并在连续漏检时逐帧衰减。
    "max_velocity_x_px_s": 720.0,
    "max_velocity_y_px_s": 560.0,
    "coast_velocity_decay": 0.96,

    # 控制中心前瞻位移限制。
    "max_lead_x_px": 55.0,
    "max_lead_y_px": 42.0,

    "initial_pos_var": 90.0,
    "initial_vel_var": 120000.0,
}

CONTROL_DT_MAX = 0.180

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


def safe_float(value, default=0.0):
    """兼容部分CanMV对象偶发返回None，防止单个坏值终止整个闭环。"""
    try:
        result = float(value)
        return float(default) if result != result else result
    except Exception:
        return float(default)


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

    def __init__(self, name, step_pin, pwm_channel, dir_pin, positive_dir_level, reverse,
                 min_hz, max_hz, accel_hz_s, decel_hz_s, soft_limit_steps):
        self.name = str(name)
        self.step_pin = int(step_pin)
        self.pwm_channel = int(pwm_channel)
        self.dir_pin_number = int(dir_pin)
        self.positive_dir_level = 1 if positive_dir_level else 0
        self.reverse = bool(reverse)

        self.min_hz = float(min_hz)
        self.max_hz = float(max_hz)
        self.accel_hz_s = float(accel_hz_s)
        self.decel_hz_s = float(decel_hz_s)
        self.soft_limit_steps = 0.0 if not SOFT_LIMITS_ENABLED else max(0.0, float(soft_limit_steps))

        self.target_hz = 0.0
        self.output_hz = 0.0       # Ramp 内部有符号频率
        self.applied_hz = 0.0      # 实际送入 PWM 的有符号频率
        self.virtual_steps = 0.0
        self.direction_sign = 0
        self.running = False
        self.last_pwm_freq = 0
        self.last_pwm_update_ms = time.ticks_ms()
        self.limit_hit = 0
        self.enabled = True
        self.pwm_probe_printed = False

        self.dir_pin = make_gpio_output(self.dir_pin_number, self.positive_dir_level)
        # CanMV v1.4.x使用“FPIOA映射 + PWM通道号”，不能把GPIO42/43
        # 直接当作PWM通道号。旧写法可创建对象，但未必有实际引脚输出。
        self.pwm_fpioa = FPIOA()
        self.pwm_fpioa.set_function(
            self.step_pin, self.pwm_fpioa.PWM0 + self.pwm_channel)
        # Yahboom CanMV v1.4.3的PWM构造器不接受enable=参数，也不接受
        # 第4个位置参数。FPIOA已显式映射时，传PWM通道号即可。
        try:
            self.pwm = PWM(
                self.pwm_channel, freq=int(PWM_INIT_FREQ_HZ), duty=0)
        except TypeError:
            self.pwm = PWM(self.pwm_channel, int(PWM_INIT_FREQ_HZ), 0)
        self._set_direction(1)
        self._stop_pwm()
        limit_deg = (
            self.soft_limit_steps * 360.0 / float(PULSES_PER_REV)
            if self.soft_limit_steps > 0.0 else 0.0
        )
        print(
            "[%s] STEP GPIO%d/PWM%d / DIR GPIO%d / min=%.0f max=%.0fHz "
            "limit=%.0f steps (±%.1f deg)" % (
                self.name, self.step_pin, self.pwm_channel, self.dir_pin_number,
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
        try:
            self.pwm.enable(False)
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
        requested_freq = max(1, int(round(mag)))
        now = time.ticks_ms()
        delta = abs(requested_freq - self.last_pwm_freq)
        # 细调区也至少相差2Hz才重装定时器；很小的变化最多延迟70ms同步，
        # 避免每帧1Hz波动反复打断硬件脉冲列。
        min_delta = max(2, int(round(requested_freq * PWM_FREQ_REL_DELTA)))
        refresh_due = ticks_diff_ms(now, self.last_pwm_update_ms) >= int(PWM_FREQ_MAX_HOLD_MS)
        if self.last_pwm_freq <= 0 or delta >= min_delta or (delta > 0 and refresh_due):
            self.pwm.freq(requested_freq)
            self.last_pwm_freq = requested_freq
            self.last_pwm_update_ms = now
        if not self.running:
            self.pwm.duty(int(STEP_DUTY_PERCENT))
            try:
                self.pwm.enable(True)
            except Exception:
                pass
            self.running = True
        # virtual_steps 必须按实际已装入硬件的频率累计，而不是未应用的请求值。
        self.applied_hz = float(s * self.last_pwm_freq)
        return self.applied_hz

    def _soft_limit_allows(self, signed_hz):
        self.limit_hit = 0
        if (not SOFT_LIMITS_ENABLED) or self.soft_limit_steps <= 0.0:
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

    def move_pulses(self, signed_pulses, frequency_hz, max_run_ms=1500):
        """用硬件PWM执行有限脉冲；CanMV无计数反馈时按实测通电时间估算脉冲数。"""
        requested = int(round(float(signed_pulses)))
        direction = sign_of(requested)
        count = abs(requested)
        if direction == 0 or count <= 0 or not self.enabled:
            self.hard_stop()
            return (0, 0, 0)
        count = min(count, 100)
        hz = clamp(abs(float(frequency_hz)), self.min_hz, self.max_hz)
        max_by_time = max(1, int(hz * max(1, int(max_run_ms)) / 1000.0))
        count = min(count, max_by_time)
        if SOFT_LIMITS_ENABLED and self.soft_limit_steps > 0.0:
            if direction > 0:
                remaining = int(math.floor(self.soft_limit_steps - self.virtual_steps))
            else:
                remaining = int(math.floor(self.soft_limit_steps + self.virtual_steps))
            count = min(count, max(0, remaining))
        if count <= 0:
            self.limit_hit = direction
            self.hard_stop()
            return (0, 0, 0)

        self.hard_stop()
        self._set_direction(direction)
        requested_freq = max(1, int(round(hz)))
        self.pwm.freq(requested_freq)
        self.last_pwm_freq = requested_freq
        self.last_pwm_update_ms = time.ticks_ms()
        duration_us = max(1, int(round(count * 1000000.0 / requested_freq)))
        start_us = time.ticks_us()
        self.pwm.duty(int(STEP_DUTY_PERCENT))
        try:
            self.pwm.enable(True)
        except Exception:
            pass
        self.running = True
        self.applied_hz = float(direction * requested_freq)
        if not self.pwm_probe_printed:
            self.pwm_probe_printed = True
            try:
                print("PWM_ACTIVE %s GPIO%d PWM%d freq=%s duty=%s" % (
                    self.name, self.step_pin, self.pwm_channel,
                    str(self.pwm.freq()), str(self.pwm.duty())))
            except Exception as e:
                print("PWM_ACTIVE %s readback unavailable: %s" % (self.name, str(e)))
        try:
            time.sleep_us(duration_us)
        finally:
            self._stop_pwm()
        elapsed_us = max(1, time.ticks_diff(time.ticks_us(), start_us))
        estimated = max(1, int(round(elapsed_us * requested_freq / 1000000.0)))
        # 不把估算值伪装成编码器；安全累计使用较保守的较大者。
        accounted = max(count, estimated)
        self.virtual_steps += direction * accounted
        if SOFT_LIMITS_ENABLED and self.soft_limit_steps > 0.0:
            self.virtual_steps = clamp(
                self.virtual_steps, -self.soft_limit_steps, self.soft_limit_steps
            )
        self.output_hz = 0.0
        self.target_hz = 0.0
        return (direction * count, direction * estimated, elapsed_us)

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
                # TRACK时将动态ROI复制成连续RGB888图像后再调用cv_lite，
                # 避免ulab非连续切片的步长兼容问题，也避免全帧Canny/轮廓计算。
                cv_np = img_np
                cv_shape = IMAGE_SHAPE
                cv_offset_x = 0
                cv_offset_y = 0
                true_roi_used = False
                roi_img = None
                roi_ratio = (roi_w * roi_h) / float(DETECT_WIDTH * DETECT_HEIGHT)
                if (
                    bool(RECT_CFG["cv_true_roi_enable"])
                    and not self.last_full_scan
                    and roi_ratio <= float(RECT_CFG["cv_true_roi_max_area_ratio"])
                ):
                    try:
                        roi_img = img.to_rgb888(roi=(roi_x, roi_y, roi_w, roi_h))
                        cv_np = roi_img.to_numpy_ref()
                        cv_shape = [int(cv_np.shape[0]), int(cv_np.shape[1])]
                        cv_offset_x = roi_x
                        cv_offset_y = roi_y
                        true_roi_used = True
                    except Exception as roi_error:
                        RECT_CFG["cv_true_roi_enable"] = False
                        print("cv_lite true ROI disabled, fallback full frame:", roi_error)

                raw_cv = cv_lite.rgb888_find_rectangles_with_corners(
                    cv_shape, cv_np,
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
                    c = self.raw_candidate_from_cv(raw, cv_offset_x, cv_offset_y)
                    if c is None:
                        continue
                    ccx, ccy = c["center_det"]
                    if (
                        true_roi_used
                        or
                        self.last_full_scan
                        or (
                            roi_x <= ccx <= roi_x1
                            and roi_y <= ccy <= roi_y1
                        )
                    ):
                        candidates.append(c)
                self.last_detector_mode = "CV_ROI" if true_roi_used else "CV"
                roi_img = None
                cv_np = None
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
                    abs(cx - dx) < 5.0 * DETECT_SCALE
                    and abs(cy - dy) < 5.0 * DETECT_SCALE
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
            # 第一次捕获没有历史轨迹时，质量相近的静止靶优先选取画面中心附近者。
            if not self.ever_locked:
                cx, cy = c["center"]
                nx = (cx - CAMERA_WIDTH * 0.5) / max(1.0, CAMERA_WIDTH * 0.5)
                ny = (cy - CAMERA_HEIGHT * 0.5) / max(1.0, CAMERA_HEIGHT * 0.5)
                score -= 0.10 * min(1.5, math.sqrt(nx * nx + ny * ny))
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
            elif quality < RECT_CFG["quality_high"]:
                continue

            cx, cy = c["center"]
            dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist > gate_px:
                continue

            # 高速运动时图像会短暂模糊，内外框和对比度评分可能下降。
            # 候选若紧跟Kalman预测位置，可用时序连续性代替单帧结构评分；
            # 超出紧门限时仍必须满足原有结构条件。
            temporal_close = dist <= float(TRACK_CFG["temporal_override_px"])
            if low_stage:
                structural_ok = (
                    c["ring_score"] >= RECT_CFG["low_stage_min_ring"]
                    or c["contrast"] >= RECT_CFG["min_contrast"]
                )
                if (not structural_ok) and (not temporal_close):
                    continue
            elif self.ever_locked and not reacquire:
                structural_ok = (
                    c.get("ring_score", 0.0) >= 0.10
                    or c.get("contrast", 0.0) >= 7.0
                    or quality >= 0.52
                )
                if (not structural_ok) and (not temporal_close):
                    continue

            if self.last_candidate is not None:
                lx, ly = self.last_candidate.get("center", (px, py))
                jump = math.sqrt((cx - lx) ** 2 + (cy - ly) ** 2)
                jump_limit = float(TRACK_CFG["jump_guard_px"])
                if reacquire:
                    jump_limit *= 1.25
                if jump > jump_limit:
                    structurally_valid = (
                        c.get("ring_score", 0.0) >= TRACK_CFG["jump_guard_ring"]
                        or c.get("contrast", 0.0) >= TRACK_CFG["jump_guard_contrast"]
                    )
                    if (
                        (not temporal_close)
                        and ((quality < TRACK_CFG["jump_guard_quality"]) or (not structurally_valid))
                    ):
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
            # 搜索/捕获阶段保持现场验证过的原生角点路径；
            # 跟踪器真正锁定后才用亚像素细化，避免它干扰首次捕获。
            or not self.ever_locked
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
        # v4.5保留亚像素中心；绘制和串口显示时再按需要取整。
        return (
            float(clamp(x, 0.0, CAMERA_WIDTH - 1.0)),
            float(clamp(y, 0.0, CAMERA_HEIGHT - 1.0)),
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
                # 低置信候选立即精修；高置信COAST恢复仍遵循精修间隔，
                # 避免刚恢复检测时由cornerSubPix额外耗时造成帧周期尖峰。
                force=bool(low_used),
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
            "Final v4.7 tuning: stronger X dynamic tracking; vision thresholds kept strict",
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

# ============================================================
# 17. v4.0：D36A步进云台 + 矩形中心/激光点闭环整合
# ============================================================
# 关键原则：
# 1) 执行器继续使用 D36A STEP/DIR，绝不使用 50Hz 舵机角度 PWM；
# 2) 矩形检测、Kalman、候选关联完全继承上面的 v4.7 稳定版本；
# 3) 新控制误差 = 矩形中心 - 激光点中心；
# 4) 摄像头、矩形、激光任一未就绪时，STEP PWM 占空比保持 0；
# 5) GPIO35 只控制激光 TTL/EN，D36A EN1/EN2 仍固定接驱动板 5V。

LASER_IO_PIN = 35
LASER_ACTIVE_LEVEL = 1
LASER_CFG = {
    # 两级LAB阈值：
    # 1) 未锁定时只允许“严格红色核心”参与捕获，避免黑白矩形边缘和彩色噪声被当成激光；
    # 2) 已锁定后如果严格核心暂时消失，才允许在预测位置附近使用较宽的红色光晕阈值续跟。
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

    # 检测尺度下的几何筛选；面积和长度由320x240基准自动缩放。
    # 激光点应当小、紧凑；大色块、细长边缘和稀疏噪声直接剔除。
    "pixels_min": 2,
    "area_min": 2,
    "area_max": int(round(360 * DETECT_AREA_SCALE)),
    "target_area": 10.0 * DETECT_AREA_SCALE,
    "max_w_det": 32.0 * DETECT_SCALE_X,
    "max_h_det": 32.0 * DETECT_SCALE_Y,
    "max_aspect": 4.2,
    "min_density": 0.07,
    "merge_margin": 0,

    # 时序门控。锁定后的门限必须明显小于旧版42px，防止从一个红噪声跳到另一个。
    "gate_locked_px_det": 24.0 * DETECT_SCALE,
    "gate_reacquire_px_det": 150.0 * DETECT_SCALE,
    "acquire_jump_px_det": 40.0 * DETECT_SCALE,
    "max_lost_frames": 18,
    "acquire_frames": 1,

    # 未锁定时的歧义保护：候选过多或第一、第二名过于接近时不允许闭环启动。
    "max_acquire_candidates": 6,
    "min_acquire_score": 0.38,
    "min_locked_score": 0.22,
    "ambiguity_margin": 0.08,

    "position_alpha": 0.72,
    "velocity_alpha": 0.25,
    "velocity_limit_logical": 1000.0,

    # RGB565局部红色加权质心，用于替代整数blob中心。
    "subpixel_radius": max(3, int(round(5 * DETECT_SCALE))),
    "subpixel_red_floor": 70,
    "subpixel_red_excess": 12,
    "subpixel_max_shift_det": 4.0 * DETECT_SCALE,

    # 丢锁后保留最后可靠位置作为重捕获锚点，电机已停时真实光点不应远跳。
    "anchor_keep_frames": 45,
    "anchor_gate_det": 30.0 * DETECT_SCALE,
    "dynamic_gate_max_det": 60.0 * DETECT_SCALE,
    "halo_gate_scale": 0.78,
    "locked_area_ratio_max": 7.0,

    # 优先只在目标矩形及其附近寻找激光，避免全画面红色干扰。
    # 连续找不到时才周期性回退到全画面严格核心搜索。
    "full_scan_when_unlocked": False,
    "rect_roi_margin_det": 42.0 * DETECT_SCALE,
    "roi_margin_det": 20.0 * DETECT_SCALE,
    "full_scan_fallback_frames": 8,
}


class LaserTTL35:
    """GPIO35只输出逻辑电平，不为激光模块供电。"""

    def __init__(self, pin_no=LASER_IO_PIN, active_level=LASER_ACTIVE_LEVEL):
        self.pin_no = int(pin_no)
        self.active_level = 1 if active_level else 0
        self.pin = make_gpio_output(self.pin_no, 1 - self.active_level)
        self.enabled = False
        self.off()
        print("Laser TTL GPIO%d ready active=%d default=OFF" % (
            self.pin_no, self.active_level
        ))

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
        """在线切换高/低电平有效。切换后保持原来的开关命令状态。"""
        was_enabled = bool(self.enabled)
        self.active_level = 1 if float(level) >= 0.5 else 0
        self.set(was_enabled)
        print("Laser TTL active level changed to %d; commanded_on=%d" % (
            self.active_level, 1 if self.enabled else 0
        ))
        return self.active_level

    def deinit(self):
        try:
            self.off()
        except Exception:
            pass
        try:
            self.pin.deinit()
        except Exception:
            pass


class LaserSpotTrackerScaled:
    """在低延迟RGB565图像上识别红色激光，输出640x480逻辑坐标。

    v8.0关键策略：
    - 未锁定：只使用严格CORE阈值；静止靶模式允许高质量首帧锁定；
    - 已锁定：CORE优先，CORE缺失时只在预测位置附近使用HALO阈值；
    - 候选过多或最优候选不突出时判为歧义，不给电机控制；
    - 检测器可输出短时预测用于显示，但静止靶控制器不会把预测位置当真实测量。
    """

    def __init__(self):
        self.position_det = None
        self.position = None
        self.provisional_det = None
        self.velocity_det = (0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.locked = False
        self.acquire_count = 0
        self.lost_frames = 0
        self.confidence = 0.0
        self.candidate_count = 0
        self.raw_blob_count = 0
        self.core_blob_count = 0
        self.halo_blob_count = 0
        self.ambiguous = 0
        self.reject_area = 0
        self.reject_shape = 0
        self.reject_density = 0
        self.reject_gate = 0
        self.last_area = 0.0
        self.last_density = 0.0
        self.last_roi = (0, 0, DETECT_WIDTH, DETECT_HEIGHT)
        self.find_error_count = 0
        self.last_find_error = ""
        self.frame_index = 0
        self.last_candidates_det = []
        self.last_source = "NONE"
        self.last_reliable_det = None
        self.anchor_age = 9999
        self.subpixel_offset_det = (0.0, 0.0)

    def reset(self):
        self.position_det = None
        self.position = None
        self.provisional_det = None
        self.velocity_det = (0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.locked = False
        self.acquire_count = 0
        self.lost_frames = 0
        self.confidence = 0.0
        self.candidate_count = 0
        self.raw_blob_count = 0
        self.core_blob_count = 0
        self.halo_blob_count = 0
        self.ambiguous = 0
        self.reject_area = 0
        self.reject_shape = 0
        self.reject_density = 0
        self.reject_gate = 0
        self.last_area = 0.0
        self.last_density = 0.0
        self.last_candidates_det = []
        self.last_source = "NONE"
        self.last_reliable_det = None
        self.anchor_age = 9999
        self.subpixel_offset_det = (0.0, 0.0)

    @staticmethod
    def _blob_value(blob, method_name, tuple_index, default=0):
        try:
            value = getattr(blob, method_name)()
            return default if value is None else value
        except Exception:
            try:
                value = blob[tuple_index]
                return default if value is None else value
            except Exception:
                return default

    @staticmethod
    def _clip_roi(x0, y0, x1, y1):
        x0 = int(clamp(math.floor(x0), 0, DETECT_WIDTH - 1))
        y0 = int(clamp(math.floor(y0), 0, DETECT_HEIGHT - 1))
        x1 = int(clamp(math.ceil(x1), x0 + 1, DETECT_WIDTH))
        y1 = int(clamp(math.ceil(y1), y0 + 1, DETECT_HEIGHT))
        return x0, y0, x1 - x0, y1 - y0

    @staticmethod
    def core_threshold():
        return (
            int(LASER_CFG["core_l_min"]), int(LASER_CFG["core_l_max"]),
            int(LASER_CFG["core_a_min"]), int(LASER_CFG["core_a_max"]),
            int(LASER_CFG["core_b_min"]), int(LASER_CFG["core_b_max"]),
        )

    @staticmethod
    def halo_threshold():
        return (
            int(LASER_CFG["halo_l_min"]), int(LASER_CFG["halo_l_max"]),
            int(LASER_CFG["halo_a_min"]), int(LASER_CFG["halo_a_max"]),
            int(LASER_CFG["halo_b_min"]), int(LASER_CFG["halo_b_max"]),
        )

    def _build_roi(self, rect_box_det):
        # 已锁定时只在激光预测位置附近搜索，矩形误检不能把搜索窗口拉到远处红点。
        if self.locked and self.position_det is not None:
            speed = math.sqrt(self.velocity_det[0] ** 2 + self.velocity_det[1] ** 2)
            g = clamp(
                float(LASER_CFG["gate_locked_px_det"]) + speed * 0.05 + 5.0 * DETECT_SCALE,
                20.0 * DETECT_SCALE, float(LASER_CFG["dynamic_gate_max_det"]),
            )
            px = self.position_det[0]
            py = self.position_det[1]
            return self._clip_roi(px - g, py - g, px + g, py + g)

        boxes = []
        anchor = self.provisional_det
        if anchor is None and self.anchor_age <= int(LASER_CFG["anchor_keep_frames"]):
            anchor = self.last_reliable_det
        if anchor is not None:
            g = float(LASER_CFG["anchor_gate_det"])
            boxes.append((anchor[0] - g, anchor[1] - g, anchor[0] + g, anchor[1] + g))

        if rect_box_det is not None:
            x, y, w, h = rect_box_det
            m = float(LASER_CFG["rect_roi_margin_det"])
            boxes.append((x - m, y - m, x + w + m, y + h + m))

        fallback = self.anchor_age > int(LASER_CFG["full_scan_fallback_frames"])
        if LASER_CFG["full_scan_when_unlocked"] or fallback or not boxes:
            return 0, 0, DETECT_WIDTH, DETECT_HEIGHT

        return self._clip_roi(
            min(v[0] for v in boxes), min(v[1] for v in boxes),
            max(v[2] for v in boxes), max(v[3] for v in boxes),
        )

    def _find(self, img, threshold, roi):
        try:
            return img.find_blobs(
                [threshold], roi=roi, x_stride=1, y_stride=1,
                pixels_threshold=int(LASER_CFG["pixels_min"]),
                area_threshold=int(LASER_CFG["area_min"]),
                merge=True, margin=int(LASER_CFG["merge_margin"]),
            ) or []
        except TypeError:
            return img.find_blobs(
                [threshold], roi=roi,
                pixels_threshold=int(LASER_CFG["pixels_min"]),
                area_threshold=int(LASER_CFG["area_min"]),
                merge=True, margin=int(LASER_CFG["merge_margin"]),
            ) or []

    def _subpixel_centroid_rgb565(self, img_np, cx, cy, bw, bh):
        """在blob附近计算红色加权质心；失败时退回find_blobs中心。"""
        if img_np is None:
            return float(cx), float(cy)
        radius = int(clamp(
            max(float(LASER_CFG["subpixel_radius"]), 0.65 * max(bw, bh) + 2.0),
            3.0, 10.0,
        ))
        x0 = max(0, int(math.floor(cx)) - radius)
        x1 = min(DETECT_WIDTH - 1, int(math.ceil(cx)) + radius)
        y0 = max(0, int(math.floor(cy)) - radius)
        y1 = min(DETECT_HEIGHT - 1, int(math.ceil(cy)) + radius)
        sw = 0.0
        sx = 0.0
        sy = 0.0
        red_floor = float(LASER_CFG["subpixel_red_floor"])
        excess_floor = float(LASER_CFG["subpixel_red_excess"])
        try:
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    value = int(img_np[yy, xx, 0]) | (int(img_np[yy, xx, 1]) << 8)
                    rv = ((value >> 11) & 0x1F) * (255.0 / 31.0)
                    gv = ((value >> 5) & 0x3F) * (255.0 / 63.0)
                    bv = (value & 0x1F) * (255.0 / 31.0)
                    excess = rv - max(gv, bv)
                    if rv < red_floor or excess < excess_floor:
                        continue
                    weight = excess + 0.20 * max(0.0, rv - red_floor)
                    sw += weight
                    sx += weight * xx
                    sy += weight * yy
            if sw > 1e-6:
                rx = sx / sw
                ry = sy / sw
                shift = math.sqrt((rx - float(cx)) ** 2 + (ry - float(cy)) ** 2)
                if shift <= float(LASER_CFG["subpixel_max_shift_det"]):
                    return rx, ry
        except Exception:
            pass
        return float(cx), float(cy)

    def _candidate_from_blob(self, blob, source, reference, gate, img_np=None):
        bx = float(self._blob_value(blob, "cx", 5, 0))
        by = float(self._blob_value(blob, "cy", 6, 0))
        bw = float(self._blob_value(blob, "w", 2, 0))
        bh = float(self._blob_value(blob, "h", 3, 0))
        pixels = float(self._blob_value(blob, "pixels", 4, bw * bh))
        area = max(1.0, bw * bh)

        if area < LASER_CFG["area_min"] or area > LASER_CFG["area_max"]:
            self.reject_area += 1
            return None
        if bw <= 0.0 or bh <= 0.0 or bw > LASER_CFG["max_w_det"] or bh > LASER_CFG["max_h_det"]:
            self.reject_shape += 1
            return None
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if aspect > LASER_CFG["max_aspect"]:
            self.reject_shape += 1
            return None

        density = clamp(pixels / area, 0.0, 1.0)
        if density < LASER_CFG["min_density"]:
            self.reject_density += 1
            return None

        if self.locked and self.last_area > 0.0:
            area_ratio = max(area, self.last_area) / max(1.0, min(area, self.last_area))
            if area_ratio > float(LASER_CFG["locked_area_ratio_max"]):
                self.reject_area += 1
                return None

        dist = 0.0
        if reference is not None:
            dist = math.sqrt((bx - reference[0]) ** 2 + (by - reference[1]) ** 2)
            if dist > gate:
                self.reject_gate += 1
                return None

        proximity = 1.0 if reference is None else clamp(1.0 - dist / max(1.0, gate), 0.0, 1.0)
        area_score = clamp(
            1.0 - abs(area - LASER_CFG["target_area"])
            / max(4.0, LASER_CFG["target_area"] * 2.0),
            0.0, 1.0,
        )
        shape_score = clamp(1.0 - (aspect - 1.0) / max(0.1, LASER_CFG["max_aspect"] - 1.0), 0.0, 1.0)
        source_bonus = 0.14 if source == "CORE" else 0.0

        if self.locked:
            score = 0.58 * proximity + 0.14 * area_score + 0.16 * density + 0.12 * shape_score + source_bonus
        elif reference is not None:
            score = 0.38 * proximity + 0.20 * area_score + 0.24 * density + 0.18 * shape_score + source_bonus
        else:
            score = 0.30 * area_score + 0.40 * density + 0.30 * shape_score + source_bonus

        # 先用find_blobs中心完成候选筛选；只对最终胜出候选做逐像素亚像素质心，
        # 避免对每个CORE/HALO噪声blob运行Python双层像素循环。
        return (score, (bx, by), area, density, source, dist, bw, bh)

    def _handle_no_measurement(self, dt=0.0):
        dt = clamp(safe_float(dt, 0.033), 0.004, CONTROL_DT_MAX)
        self.lost_frames += 1
        self.anchor_age += 1
        self.confidence = 0.0
        self.last_source = "NONE"

        if (
            self.locked
            and self.position_det is not None
            and self.lost_frames <= int(LASER_PREDICT_MAX_FRAMES)
        ):
            decay = float(LASER_PREDICT_VEL_DECAY)
            self.velocity_det = (
                self.velocity_det[0] * decay,
                self.velocity_det[1] * decay,
            )
            self.position_det = (
                clamp(self.position_det[0] + self.velocity_det[0] * max(0.0, dt),
                      0.0, DETECT_WIDTH - 1.0),
                clamp(self.position_det[1] + self.velocity_det[1] * max(0.0, dt),
                      0.0, DETECT_HEIGHT - 1.0),
            )
            self.position = (
                detect_to_logical_x(self.position_det[0]),
                detect_to_logical_y(self.position_det[1]),
            )
            self.velocity = (
                self.velocity_det[0] * DETECT_TO_LOGICAL_X,
                self.velocity_det[1] * DETECT_TO_LOGICAL_Y,
            )
            self.last_source = "PREDICT"
            return self.position

        if self.locked and self.lost_frames > int(LASER_CFG["max_lost_frames"]):
            # 不清除最后可靠锚点。电机已停止，真实激光应仍在附近。
            if self.position_det is not None:
                self.last_reliable_det = self.position_det
                self.anchor_age = 0
            self.locked = False
            self.position_det = None
            self.position = None
            self.provisional_det = None
            self.acquire_count = 0
            self.velocity_det = (0.0, 0.0)
            self.velocity = (0.0, 0.0)
        return None

    def detect(self, img, img_np, rect_box_det, desired_center_logical, dt):
        dt = clamp(safe_float(dt, 0.033), 0.004, CONTROL_DT_MAX)
        self.frame_index += 1
        self.ambiguous = 0
        self.reject_area = 0
        self.reject_shape = 0
        self.reject_density = 0
        self.reject_gate = 0

        roi = self._build_roi(rect_box_det)
        self.last_roi = roi
        predicted_det = None
        if self.position_det is not None:
            predicted_det = (
                self.position_det[0] + self.velocity_det[0] * dt,
                self.position_det[1] + self.velocity_det[1] * dt,
            )

        # 未锁定时优先使用临时位置，其次使用最后可靠锚点。
        if self.locked:
            reference = predicted_det
            speed = math.sqrt(self.velocity_det[0] ** 2 + self.velocity_det[1] ** 2)
            gate = clamp(
                float(LASER_CFG["gate_locked_px_det"]) + speed * dt * 1.2,
                float(LASER_CFG["gate_locked_px_det"]),
                float(LASER_CFG["dynamic_gate_max_det"]),
            )
        else:
            reference = self.provisional_det
            if reference is None and self.anchor_age <= int(LASER_CFG["anchor_keep_frames"]):
                reference = self.last_reliable_det
            gate = float(LASER_CFG["anchor_gate_det"] if reference is not None else LASER_CFG["gate_reacquire_px_det"])

        # 稳态跟踪先只做严格CORE搜索。v6.2每帧同时运行CORE和HALO，
        # 实测两者经常返回同一光斑，使锁定后FPS从约35降到18~20。
        try:
            core_blobs = self._find(img, self.core_threshold(), roi)
            self.last_find_error = ""
        except Exception as e:
            core_blobs = []
            self.find_error_count += 1
            self.last_find_error = str(e)

        core_candidates = []
        for blob in core_blobs:
            c = self._candidate_from_blob(blob, "CORE", reference, gate, img_np)
            if c is not None:
                core_candidates.append(c)

        # CORE没有有效候选时同帧回退HALO；上一帧依赖HALO/PREDICT时也补做一次，
        # 确保暗光斑、短时丢失和恢复阶段不降低鲁棒性。
        core_score_floor = float(
            LASER_CFG["min_locked_score"]
            if self.locked else LASER_CFG["min_acquire_score"]
        )
        core_usable = any(item[0] >= core_score_floor for item in core_candidates)
        need_halo = (
            (not core_usable)
            or self.lost_frames > 0
            or self.last_source in ("HALO", "PREDICT")
        )
        halo_blobs = []
        halo_candidates = []
        halo_ref = self.position_det
        if halo_ref is None and self.anchor_age <= int(LASER_CFG["anchor_keep_frames"]):
            halo_ref = self.last_reliable_det
        if need_halo and halo_ref is not None:
            base_g = float(LASER_CFG["gate_locked_px_det"])
            g = clamp(base_g * 1.20, 18.0 * DETECT_SCALE, float(LASER_CFG["dynamic_gate_max_det"]))
            hx, hy = halo_ref
            halo_roi = self._clip_roi(hx - g, hy - g, hx + g, hy + g)
            try:
                halo_blobs = self._find(img, self.halo_threshold(), halo_roi)
            except Exception as e:
                self.find_error_count += 1
                self.last_find_error = str(e)

        for blob in halo_blobs:
            halo_gate = gate * float(LASER_CFG["halo_gate_scale"])
            c = self._candidate_from_blob(blob, "HALO", reference, halo_gate, img_np)
            if c is not None:
                halo_candidates.append(c)

        self.core_blob_count = len(core_blobs)
        self.halo_blob_count = len(halo_blobs)
        self.raw_blob_count = self.core_blob_count + self.halo_blob_count

        if self.last_find_error and (self.find_error_count <= 3 or self.frame_index % 60 == 0):
            print("laser find_blobs error on RGB565 channel:", self.last_find_error)

        candidates = core_candidates + halo_candidates

        # CORE与HALO可能描述同一光斑，按3像素邻域去重，优先保留得分更高者。
        candidates.sort(key=lambda item: item[0], reverse=True)
        dedup = []
        for item in candidates:
            px, py = item[1]
            duplicate = False
            for kept in dedup:
                kx, ky = kept[1]
                if (px - kx) ** 2 + (py - ky) ** 2 <= (3.0 * DETECT_SCALE) ** 2:
                    duplicate = True
                    break
            if not duplicate:
                dedup.append(item)
        candidates = dedup
        candidates.sort(key=lambda item: item[0], reverse=True)
        self.candidate_count = len(candidates)
        self.last_candidates_det = [
            (float(item[1][0]), float(item[1][1]), float(item[2]), float(item[3]))
            for item in candidates
        ]

        if not candidates:
            self.acquire_count = 0 if not self.locked else self.acquire_count
            return self._handle_no_measurement(dt)

        # 未锁定时，候选过多且最优者不明显，判为歧义，不启动电机。
        if not self.locked:
            top_score = float(candidates[0][0])
            second_score = float(candidates[1][0]) if len(candidates) > 1 else -1.0
            too_many = len(candidates) > int(LASER_CFG["max_acquire_candidates"])
            too_close = len(candidates) > 1 and (top_score - second_score) < float(LASER_CFG["ambiguity_margin"])
            if top_score < float(LASER_CFG["min_acquire_score"]) or (too_many and too_close):
                self.ambiguous = 1
                self.acquire_count = 0
                self.provisional_det = None
                return self._handle_no_measurement(dt)
        elif candidates[0][0] < float(LASER_CFG["min_locked_score"]):
            return self._handle_no_measurement(dt)

        score, measured_det, area, density, source, _, blob_w, blob_h = candidates[0]
        raw_det = measured_det
        if (not self.locked) or (self.frame_index % 2 == 0):
            measured_det = self._subpixel_centroid_rgb565(
                img_np, raw_det[0], raw_det[1], blob_w, blob_h
            )
            self.subpixel_offset_det = (
                measured_det[0] - raw_det[0],
                measured_det[1] - raw_det[1],
            )
        else:
            # 稳态锁定时隔帧复用最近亚像素偏移，减少Python逐像素循环，
            # 同时避免直接回退到整数blob中心引入量化抖动。
            measured_det = (
                clamp(raw_det[0] + self.subpixel_offset_det[0], 0.0, DETECT_WIDTH - 1.0),
                clamp(raw_det[1] + self.subpixel_offset_det[1], 0.0, DETECT_HEIGHT - 1.0),
            )
        self.last_area = area
        self.last_density = density
        self.confidence = clamp(score, 0.0, 1.0)
        self.last_source = source
        self.lost_frames = 0
        self.anchor_age = 0

        # 静止靶模式允许高质量首帧锁定。若以后把acquire_frames调大，
        # 仍保留原有连续位置一致性检查。
        if not self.locked:
            if self.provisional_det is None:
                self.provisional_det = measured_det
                self.acquire_count = 1
            else:
                jump = math.sqrt(
                    (measured_det[0] - self.provisional_det[0]) ** 2
                    + (measured_det[1] - self.provisional_det[1]) ** 2
                )
                if jump > float(LASER_CFG["acquire_jump_px_det"]):
                    self.provisional_det = measured_det
                    self.acquire_count = 1
                    return None

                pa = 0.55
                self.provisional_det = (
                    self.provisional_det[0] + pa * (measured_det[0] - self.provisional_det[0]),
                    self.provisional_det[1] + pa * (measured_det[1] - self.provisional_det[1]),
                )
                self.acquire_count += 1
            if self.acquire_count < int(LASER_CFG["acquire_frames"]):
                return None

            self.locked = True
            self.position_det = self.provisional_det
            self.position = (
                detect_to_logical_x(self.position_det[0]),
                detect_to_logical_y(self.position_det[1]),
            )
            self.velocity_det = (0.0, 0.0)
            self.velocity = (0.0, 0.0)
            self.last_reliable_det = self.position_det
            self.anchor_age = 0
            return self.position

        # 已锁定：位置与速度滤波。
        if self.position_det is None:
            filtered_det = measured_det
            new_velocity_det = (0.0, 0.0)
        else:
            a = float(LASER_CFG["position_alpha"])
            filtered_det = (
                self.position_det[0] + a * (measured_det[0] - self.position_det[0]),
                self.position_det[1] + a * (measured_det[1] - self.position_det[1]),
            )
            raw_vx = (filtered_det[0] - self.position_det[0]) / max(0.004, dt)
            raw_vy = (filtered_det[1] - self.position_det[1]) / max(0.004, dt)
            va = float(LASER_CFG["velocity_alpha"])
            new_velocity_det = (
                self.velocity_det[0] + va * (raw_vx - self.velocity_det[0]),
                self.velocity_det[1] + va * (raw_vy - self.velocity_det[1]),
            )

        logical_limit = float(LASER_CFG["velocity_limit_logical"])
        new_velocity_logical = (
            clamp(new_velocity_det[0] * DETECT_TO_LOGICAL_X, -logical_limit, logical_limit),
            clamp(new_velocity_det[1] * DETECT_TO_LOGICAL_Y, -logical_limit, logical_limit),
        )
        self.velocity_det = (
            new_velocity_logical[0] / DETECT_TO_LOGICAL_X,
            new_velocity_logical[1] / DETECT_TO_LOGICAL_Y,
        )
        self.velocity = new_velocity_logical
        self.position_det = filtered_det
        self.provisional_det = filtered_det
        self.last_reliable_det = filtered_det
        self.anchor_age = 0
        self.position = (
            detect_to_logical_x(filtered_det[0]),
            detect_to_logical_y(filtered_det[1]),
        )
        return self.position

# ============================================================
# v9.0 IBVS执行入口：硬件/视觉基础层之上使用全新模块化控制架构
# ============================================================
CFG = {
    "target_confirm_frames": 3, "laser_confirm_frames": 3,
    "max_measurement_time_skew_ms": 12,
    # RectangleTracker的现场验证高质量门槛是0.38。再叠加0.45会把
    # 0.38~0.45的已选主候选全部拒绝，导致永久停在SEARCH_TARGET。
    "target_min_conf": 0.38, "laser_min_conf": 0.45,
    # 当前CanMV 1.4.3的该传感器驱动明确报告不支持这三个接口。
    # 默认不反复调用；以后换成支持的固件时可改为True。
    "camera_manual_lock_enable": False,
    # 所有捕获/验证态都有重置或退回出口，不允许永久死等。
    "search_rearm_ms": 2500, "lock_target_timeout_ms": 1200,
    "search_laser_timeout_ms": 1600, "verify_laser_timeout_ms": 1800,
    "fallback_min_interval_ms": 1200, "verify_miss_tolerance": 5,
    "target_predict_frames_setup": 3,
    "target_stability_var": 1.20, "laser_stability_var": 1.00,
    "cal_x_pulses": 24, "cal_y_pulses": 24, "cal_hz": 120,
    "cal_settle_ms": 60, "cal_samples": 3, "cal_min_shift_px": 2.0,
    "jacobian_det_min": 1e-5, "jacobian_min_sine": 0.12,
    "coarse_gain": 0.80, "coarse_gain_min_ratio": 0.28,
    "coarse_near_px": 12.0, "coarse_far_px": 75.0,
    "coarse_max_pulses_x": 24, "coarse_max_pulses_y": 24,
    "coarse_pulse_hz": 260.0, "coarse_settle_ms": 28,
    "fine_enter_px": 12.0, "fine_exit_px": 18.0,
    "x_max_hz": 900.0, "y_max_hz": 760.0,
    "x_min_hz": 12.0, "y_min_hz": 10.0,
    "x_accel_hz_s": 3600.0, "y_accel_hz_s": 3200.0,
    "x_decel_hz_s": 6200.0, "y_decel_hz_s": 5600.0,
    "fine_gain": 0.70, "fine_max_pulses_x": 20,
    "fine_max_pulses_y": 20, "fine_hz": 100.0,
    "fine_settle_ms": 40, "fine_no_improve_limit": 3,
    "lock_x_px": 1.5, "lock_y_px": 1.5,
    "unlock_x_px": 3.0, "unlock_y_px": 3.0,
    "lock_confirm_frames": 10, "lock_variance_max": 0.70,
    "jacobian_update_enable": True, "jacobian_beta": 0.25,
    "jacobian_min_action_pulses": 3.0,
    "jacobian_max_element_change_ratio": 0.35,
    "jacobian_max_initial_ratio": 3.0,
    "recalibrate_target_shift_px": 8.0,
    "max_visual_miss_frames": 2, "max_reacquire_ms": 1500,
    "max_axis_run_ms": 1500, "max_single_action_pulses": 100,
    "max_total_action_pulses": 8000.0, "max_fault_retries": 3,
    "red_min": 68, "red_excess_min": 20, "red_halo_min": 52,
    "red_halo_excess": 12, "red_peak_separation": 7,
    "laser_roi_margin": 42, "laser_anchor_gate": 35,
    "laser_max_candidates": 6, "laser_ambiguity_margin": 0.08,
    "plot_interval_ms": 50, "vision_interval_ms": 500,
    "diag_interval_ms": 1000,
    # 启动时每轴16脉冲正反微动，净位移为0；用于立即区分
    # 软件状态机问题和D36A供电/EN/STEP接线问题。
    "startup_motor_probe": True, "startup_probe_pulses": 16,
    "startup_probe_hz": 160.0,
}

IBVS_INIT = 0
IBVS_SEARCH_TARGET = 1
IBVS_LOCK_TARGET = 2
IBVS_SEARCH_LASER = 3
IBVS_VERIFY_LASER = 4
IBVS_CALIBRATE_X = 5
IBVS_CALIBRATE_Y = 6
IBVS_COARSE_AIM = 7
IBVS_FINE_AIM = 8
IBVS_LOCKED = 9
IBVS_REACQUIRE = 10
IBVS_FAULT = 11
IBVS_ESTOP = 12
IBVS_STATE_NAME = (
    "INIT", "SEARCH_TARGET", "LOCK_TARGET", "SEARCH_LASER", "VERIFY_LASER",
    "CALIBRATE_X", "CALIBRATE_Y", "COARSE_AIM", "FINE_AIM", "LOCKED",
    "REACQUIRE", "FAULT", "ESTOP",
)


def median_value(values):
    if not values:
        return 0.0
    ordered = list(values)
    ordered.sort()
    n = len(ordered)
    return float(ordered[n // 2]) if n & 1 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def median_point(points):
    return (median_value([p[0] for p in points]), median_value([p[1] for p in points]))


def point_variance(points):
    if not points:
        return 1e9
    mx = sum(p[0] for p in points) / len(points)
    my = sum(p[1] for p in points) / len(points)
    return sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in points) / len(points)


class CameraManager:
    """单RGB888通道采图；矩形与激光共享同一帧和同一时间戳。"""
    def __init__(self):
        self.sensor = None
        self.capture_ms = 0.0
        self.fallback_capture_ms = 0.0
        self.ready = False

    def start(self):
        try:
            self.sensor = Sensor(id=SENSOR_ID, width=SENSOR_INPUT_WIDTH,
                                 height=SENSOR_INPUT_HEIGHT, fps=SENSOR_FPS)
        except Exception:
            self.sensor = Sensor()
        self.sensor.reset()
        self.sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT, chn=RECT_CHANNEL)
        self.sensor.set_pixformat(Sensor.RGB888, chn=RECT_CHANNEL)
        # RGB565仅为首次捕获困难时的CORE/HALO后备，不进入常态双通道闭环。
        self.sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT, chn=LASER_CHANNEL)
        self.sensor.set_pixformat(Sensor.RGB565, chn=LASER_CHANNEL)
        MediaManager.init()
        self.sensor.run()
        time.sleep_ms(300)
        for _ in range(24):
            os.exitpoint()
            self.sensor.snapshot(chn=RECT_CHANNEL)
            time.sleep_ms(8)
        self._freeze_exposure()
        self.ready = True

    def _freeze_exposure(self):
        if not bool(CFG["camera_manual_lock_enable"]):
            print("camera AE/AWB lock skipped: unsupported by current sensor firmware")
            return
        for name in ("set_auto_exposure", "set_auto_gain", "set_auto_whitebal"):
            fn = getattr(self.sensor, name, None)
            if fn is not None:
                try:
                    fn(False)
                except Exception as e:
                    print("camera lock warning", name, e)

    def capture(self):
        start = time.ticks_us()
        img = self.sensor.snapshot(chn=RECT_CHANNEL)
        end = time.ticks_us()
        self.capture_ms = max(0.0, time.ticks_diff(end, start) / 1000.0)
        return img, img.to_numpy_ref(), end

    def capture_laser_fallback(self):
        start = time.ticks_us()
        img = self.sensor.snapshot(chn=LASER_CHANNEL)
        end = time.ticks_us()
        self.fallback_capture_ms = max(0.0, time.ticks_diff(end, start) / 1000.0)
        return img, img.to_numpy_ref(), end

    def close(self):
        if self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception:
                pass


class TargetDetector:
    def __init__(self):
        self.tracker = RectangleTracker()

    def reset(self):
        self.tracker.reset()

    def detect(self, img, img_np, dt, timestamp_us, allow_prediction=False):
        result = self.tracker.step(img, img_np, dt)
        candidate = result.get("candidate")
        source = str(result.get("source", "NONE"))
        real = candidate is not None and source.startswith("DETECT")
        predicted = bool(
            allow_prediction and source == "PREDICT" and result.get("center") is not None
            and int(result.get("miss_frames", 999)) <= int(CFG["target_predict_frames_setup"])
        )
        predicted_center = None
        if predicted:
            logical_center = result.get("center")
            predicted_center = (
                logical_to_detect_x(logical_center[0]),
                logical_to_detect_y(logical_center[1]),
            )
        confidence = (float(candidate.get("quality", 0.0)) if real
                      else float(result.get("confidence", 0.0)) if predicted else 0.0)
        candidates = result.get("candidates") or []
        ambiguous = False
        if real and len(candidates) > 1:
            a, b = candidates[0], candidates[1]
            ax, ay = a["center_det"]
            bx, by = b["center_det"]
            separated = (ax - bx) ** 2 + (ay - by) ** 2 > 12.0 ** 2
            ambiguous = separated and abs(float(a["quality"]) - float(b["quality"])) < 0.05
        # 多候选在固定跑道场景很常见（外框、内框或背景边缘）。
        # 这里不再一票否决；主候选仍要通过外层3+3帧的连续性和方差确认。
        valid = ((real and confidence >= float(CFG["target_min_conf"])) or predicted)
        best_quality = float(candidates[0].get("quality", 0.0)) if candidates else 0.0
        if predicted:
            reject_reason = "PREDICT_OK"
        elif valid:
            reject_reason = "OK_AMBIG_STABLE_GATE" if ambiguous else "OK"
        elif not candidates:
            reject_reason = "NO_CANDIDATE"
        elif not real:
            reject_reason = "TRACKER_GATE"
        elif confidence < float(CFG["target_min_conf"]):
            reject_reason = "LOW_QUALITY"
        else:
            reject_reason = "REJECTED"
        return {
            "valid": valid,
            "center": candidate["center_det"] if real else predicted_center,
            "corners": candidate.get("corners_det") if real else result.get("corners_det") if predicted else None,
            "outer_box": candidate.get("box_det") if real else result.get("box_det"),
            "inner_box": candidate.get("inner_box_det") if real else None,
            "confidence": confidence,
            "best_quality": best_quality,
            "candidate_count": len(candidates),
            "ambiguous": ambiguous,
            "predicted": predicted,
            "reject_reason": reject_reason,
            "source": source,
            "timestamp_us": int(timestamp_us),
            "raw": result,
            "detect_ms": float(result.get("tracker_ms", 0.0)),
        }


class LaserDetector:
    """同帧RGB888红色指数主检测，可靠锚点附近允许较宽光晕阈值。"""
    def __init__(self):
        self.anchor = None
        self.locked = False
        self.last_source = "NONE"
        self.detect_ms = 0.0
        self.blob_error_printed = False

    def reset(self):
        self.anchor = None
        self.locked = False
        self.last_source = "NONE"

    @staticmethod
    def _roi(target_box):
        if target_box is None:
            return (0, 0, DETECT_WIDTH, DETECT_HEIGHT)
        x, y, w, h = target_box
        margin = max(float(CFG["laser_roi_margin"]), 0.9 * max(w, h))
        x0 = int(clamp(math.floor(x - margin), 0, DETECT_WIDTH - 1))
        y0 = int(clamp(math.floor(y - margin), 0, DETECT_HEIGHT - 1))
        x1 = int(clamp(math.ceil(x + w + margin), x0 + 1, DETECT_WIDTH))
        y1 = int(clamp(math.ceil(y + h + margin), y0 + 1, DETECT_HEIGHT))
        return (x0, y0, x1 - x0, y1 - y0)

    @staticmethod
    def _rgb(img_np, x, y):
        return int(img_np[y, x, 0]), int(img_np[y, x, 1]), int(img_np[y, x, 2])

    def _peaks(self, img_np, roi, red_min, excess_min):
        x0, y0, w, h = roi
        peaks = []
        sep2 = float(CFG["red_peak_separation"]) ** 2
        for yy in range(y0, y0 + h, 2):
            for xx in range(x0, x0 + w, 2):
                r, g, b = self._rgb(img_np, xx, yy)
                excess = r - max(g, b)
                if r < red_min or excess < excess_min:
                    continue
                score = excess + 0.12 * r
                merged = False
                for item in peaks:
                    if (xx - item[1]) ** 2 + (yy - item[2]) ** 2 <= sep2:
                        if score > item[0]:
                            item[0], item[1], item[2] = score, xx, yy
                        merged = True
                        break
                if not merged:
                    peaks.append([score, xx, yy])
                    if len(peaks) > int(CFG["laser_max_candidates"]) * 2:
                        peaks.sort(key=lambda p: p[0], reverse=True)
                        peaks.pop()
        peaks.sort(key=lambda p: p[0], reverse=True)
        return peaks[:int(CFG["laser_max_candidates"])]

    def _refine(self, img_np, peak, red_min, excess_min):
        _, px, py = peak
        radius = 6
        sw = sx = sy = 0.0
        count = 0
        minx = miny = 9999
        maxx = maxy = -1
        for yy in range(max(0, py - radius), min(DETECT_HEIGHT, py + radius + 1)):
            for xx in range(max(0, px - radius), min(DETECT_WIDTH, px + radius + 1)):
                r, g, b = self._rgb(img_np, xx, yy)
                excess = r - max(g, b)
                if r < red_min or excess < excess_min:
                    continue
                weight = excess + 0.15 * max(0, r - red_min)
                sw += weight
                sx += weight * xx
                sy += weight * yy
                count += 1
                minx, maxx = min(minx, xx), max(maxx, xx)
                miny, maxy = min(miny, yy), max(maxy, yy)
        if count <= 0 or sw <= 0.0:
            return None
        area = max(1, (maxx - minx + 1) * (maxy - miny + 1))
        density = count / float(area)
        confidence = clamp(0.30 + 0.35 * min(1.0, peak[0] / 90.0) + 0.35 * density, 0.0, 1.0)
        return (confidence, (sx / sw, sy / sw), count, peak[0], density)

    def _scan(self, img, target_box, halo=False):
        roi = self._roi(target_box)
        threshold = (LaserSpotTrackerScaled.halo_threshold() if halo
                     else LaserSpotTrackerScaled.core_threshold())
        try:
            try:
                blobs = img.find_blobs(
                    [threshold], roi=roi, x_stride=1, y_stride=1,
                    pixels_threshold=int(LASER_CFG["pixels_min"]),
                    area_threshold=int(LASER_CFG["area_min"]),
                    merge=True, margin=int(LASER_CFG["merge_margin"]),
                ) or []
            except TypeError:
                blobs = img.find_blobs(
                    [threshold], roi=roi,
                    pixels_threshold=int(LASER_CFG["pixels_min"]),
                    area_threshold=int(LASER_CFG["area_min"]),
                    merge=True, margin=int(LASER_CFG["merge_margin"]),
                ) or []
        except Exception as e:
            if "IDE interrupt" in str(e):
                raise
            if not self.blob_error_printed:
                self.blob_error_printed = True
                print("RGB888 laser find_blobs unavailable; low-rate RGB565 fallback remains:", e)
            return []
        candidates = []
        for blob in blobs:
            cx = float(LaserSpotTrackerScaled._blob_value(blob, "cx", 5, 0))
            cy = float(LaserSpotTrackerScaled._blob_value(blob, "cy", 6, 0))
            bw = float(LaserSpotTrackerScaled._blob_value(blob, "w", 2, 0))
            bh = float(LaserSpotTrackerScaled._blob_value(blob, "h", 3, 0))
            pixels = float(LaserSpotTrackerScaled._blob_value(blob, "pixels", 4, bw * bh))
            area = max(1.0, bw * bh)
            if (area > float(LASER_CFG["area_max"]) or bw <= 0 or bh <= 0 or
                    bw > float(LASER_CFG["max_w_det"]) or bh > float(LASER_CFG["max_h_det"])):
                continue
            density = clamp(pixels / area, 0.0, 1.0)
            aspect = max(bw, bh) / max(1.0, min(bw, bh))
            if density < float(LASER_CFG["min_density"]) or aspect > float(LASER_CFG["max_aspect"]):
                continue
            shape_score = clamp(1.0 - (aspect - 1.0) / 3.2, 0.0, 1.0)
            area_score = clamp(1.0 - abs(area - float(LASER_CFG["target_area"])) /
                               max(4.0, 2.0 * float(LASER_CFG["target_area"])), 0.0, 1.0)
            confidence = 0.28 + 0.28 * density + 0.20 * shape_score + 0.14 * area_score
            if not halo:
                confidence += 0.10
            if self.anchor is not None:
                dx = cx - self.anchor[0]
                dy = cy - self.anchor[1]
                if dx * dx + dy * dy > float(CFG["laser_anchor_gate"]) ** 2:
                    continue
                proximity = 1.0 - math.sqrt(dx * dx + dy * dy) / max(1.0, float(CFG["laser_anchor_gate"]))
                confidence += 0.12 * clamp(proximity, 0.0, 1.0)
            candidates.append((clamp(confidence, 0.0, 1.0), (cx, cy), pixels,
                               confidence * 90.0, density))
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[:int(CFG["laser_max_candidates"])]

    def detect(self, img, img_np, target_box, timestamp_us):
        start = time.ticks_us()
        candidates = self._scan(img, target_box, halo=False)
        source = "RGB888_BLOB"
        if not candidates and self.anchor is not None:
            candidates = self._scan(img, target_box, halo=True)
            source = "HALO"
        ambiguous = False
        if len(candidates) > 1:
            a, b = candidates[0], candidates[1]
            dist2 = (a[1][0] - b[1][0]) ** 2 + (a[1][1] - b[1][1]) ** 2
            ambiguous = dist2 > 8.0 ** 2 and a[0] - b[0] < float(CFG["laser_ambiguity_margin"])
        best = candidates[0] if candidates else None
        if best is not None:
            red_min = int(CFG["red_halo_min"] if source == "HALO" else CFG["red_min"])
            excess_min = int(CFG["red_halo_excess"] if source == "HALO" else CFG["red_excess_min"])
            refined = self._refine(
                img_np, [best[3], int(round(best[1][0])), int(round(best[1][1]))],
                red_min, excess_min)
            if refined is not None:
                best = (max(best[0], refined[0]), refined[1], refined[2], refined[3], refined[4])
        valid = best is not None and best[0] >= float(CFG["laser_min_conf"]) and not ambiguous
        if valid:
            center = best[1]
            if self.anchor is not None:
                center = (self.anchor[0] + 0.72 * (center[0] - self.anchor[0]),
                          self.anchor[1] + 0.72 * (center[1] - self.anchor[1]))
            self.anchor = center
            self.locked = True
            self.last_source = source
        else:
            center = None
        self.detect_ms = perf_ms(start)
        return {
            "valid": valid, "center": center,
            "area": float(best[2]) if best else 0.0,
            "peak": float(best[3]) if best else 0.0,
            "confidence": float(best[0]) if best else 0.0,
            "candidate_count": len(candidates), "ambiguous": ambiguous,
            "source": source if best else "NONE",
            "timestamp_us": int(timestamp_us), "detect_ms": self.detect_ms,
        }

    def accept_external(self, result):
        if result and result.get("valid"):
            self.anchor = result["center"]
            self.locked = True
            self.last_source = str(result.get("source", "DIFF"))


class DifferenceLaserVerifier:
    """仅在首次捕获困难时执行的低频激光开/关红色指数差分。"""
    def __init__(self):
        self.active = False
        self.phase = 0
        self.phase_ms = 0
        self.samples = None
        self.roi = None

    def start(self, roi, laser_output):
        self.active = True
        self.phase = 0
        self.phase_ms = time.ticks_ms()
        self.samples = None
        self.roi = roi
        laser_output.off()

    def cancel(self, laser_output):
        self.active = False
        self.samples = None
        laser_output.on()

    def update(self, img_np, timestamp_us, laser_output):
        if not self.active or ticks_diff_ms(time.ticks_ms(), self.phase_ms) < 40:
            return None
        x0, y0, w, h = self.roi
        if self.phase == 0:
            samples = []
            for yy in range(y0, y0 + h, 2):
                for xx in range(x0, x0 + w, 2):
                    r = int(img_np[yy, xx, 0])
                    g = int(img_np[yy, xx, 1])
                    b = int(img_np[yy, xx, 2])
                    samples.append(r - max(g, b))
            self.samples = samples
            laser_output.on()
            self.phase = 1
            self.phase_ms = time.ticks_ms()
            return None
        sw = sx = sy = 0.0
        idx = 0
        count = 0
        for yy in range(y0, y0 + h, 2):
            for xx in range(x0, x0 + w, 2):
                r = int(img_np[yy, xx, 0])
                g = int(img_np[yy, xx, 1])
                b = int(img_np[yy, xx, 2])
                current = r - max(g, b)
                delta = current - self.samples[idx]
                idx += 1
                if delta >= 14 and r >= 55:
                    sw += delta
                    sx += delta * xx
                    sy += delta * yy
                    count += 1
        self.active = False
        self.samples = None
        if count < 1 or sw <= 0.0:
            return {"valid": False, "center": None, "area": 0.0, "peak": 0.0,
                    "confidence": 0.0, "candidate_count": 0, "ambiguous": False,
                    "source": "DIFF", "timestamp_us": int(timestamp_us), "detect_ms": 0.0}
        return {"valid": True, "center": (sx / sw, sy / sw), "area": float(count),
                "peak": 0.0, "confidence": 0.75, "candidate_count": 1,
                "ambiguous": False, "source": "DIFF",
                "timestamp_us": int(timestamp_us), "detect_ms": 0.0}


class VisualMeasurement:
    def build(self, target, laser):
        valid = bool(target.get("valid") and laser.get("valid"))
        # 矩形的多候选已由跟踪关联和多帧稳定门处理；
        # 激光多候选仍一票否决，防止向错误光点闭环。
        if laser.get("ambiguous"):
            valid = False
        skew_us = abs(int(target.get("timestamp_us", 0)) - int(laser.get("timestamp_us", 0)))
        if skew_us > int(CFG["max_measurement_time_skew_ms"]) * 1000:
            valid = False
        if not valid:
            return {"valid": False, "target": None, "laser": None, "error": None,
                    "target_conf": float(target.get("confidence", 0.0)),
                    "laser_conf": float(laser.get("confidence", 0.0)),
                    "timestamp_us": max(int(target.get("timestamp_us", 0)),
                                        int(laser.get("timestamp_us", 0))), "skew_us": skew_us}
        tx, ty = target["center"]
        lx, ly = laser["center"]
        return {"valid": True, "target": (tx, ty), "laser": (lx, ly),
                "error": (tx - lx, ty - ly),
                "target_predicted": bool(target.get("predicted")),
                "target_conf": float(target["confidence"]),
                "laser_conf": float(laser["confidence"]),
                "timestamp_us": max(int(target["timestamp_us"]), int(laser["timestamp_us"])),
                "skew_us": skew_us}


class JacobianEstimator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.j = [[0.0, 0.0], [0.0, 0.0]]
        self.initial = None
        self.valid = False
        self.update_count = 0
        self.reject_count = 0
        self.consecutive_rejects = 0

    def set_column(self, index, de, pulses):
        scale = 1.0 / max(1.0, abs(float(pulses)))
        self.j[0][index] = float(de[0]) * scale
        self.j[1][index] = float(de[1]) * scale

    def metrics(self, matrix=None):
        j = self.j if matrix is None else matrix
        j11, j12 = j[0]
        j21, j22 = j[1]
        det = j11 * j22 - j12 * j21
        n1 = math.sqrt(j11 * j11 + j21 * j21)
        n2 = math.sqrt(j12 * j12 + j22 * j22)
        sine = abs(det) / max(1e-9, n1 * n2)
        return det, n1, n2, sine

    def validate(self, matrix=None):
        det, n1, n2, sine = self.metrics(matrix)
        if n1 < 0.01 or n2 < 0.01:
            return False
        if abs(det) < float(CFG["jacobian_det_min"]) or sine < float(CFG["jacobian_min_sine"]):
            return False
        inv_max = max(abs(n1 / det), abs(n2 / det))
        return inv_max < 5000.0

    def finalize(self):
        self.valid = self.validate()
        if self.valid:
            self.initial = [[self.j[0][0], self.j[0][1]], [self.j[1][0], self.j[1][1]]]
        return self.valid

    def solve(self, error, gain=1.0, damping=0.002):
        if not self.valid:
            return (0.0, 0.0)
        j11, j12 = self.j[0]
        j21, j22 = self.j[1]
        a = j11 * j11 + j21 * j21 + damping
        b = j11 * j12 + j21 * j22
        d = j12 * j12 + j22 * j22 + damping
        det = a * d - b * b
        if abs(det) < 1e-9:
            return (0.0, 0.0)
        r1 = j11 * error[0] + j21 * error[1]
        r2 = j12 * error[0] + j22 * error[1]
        nx = -float(gain) * (d * r1 - b * r2) / det
        ny = -float(gain) * (-b * r1 + a * r2) / det
        return nx, ny

    def update(self, before, after, action, beta):
        if not self.valid:
            return False
        nx, ny = action
        denom = nx * nx + ny * ny
        if denom < float(CFG["jacobian_min_action_pulses"]) ** 2:
            return False
        de0, de1 = after[0] - before[0], after[1] - before[1]
        if de0 * de0 + de1 * de1 < 0.50 ** 2:
            return False
        pred0 = self.j[0][0] * nx + self.j[0][1] * ny
        pred1 = self.j[1][0] * nx + self.j[1][1] * ny
        candidate = [list(self.j[0]), list(self.j[1])]
        candidate[0][0] += beta * (de0 - pred0) * nx / denom
        candidate[0][1] += beta * (de0 - pred0) * ny / denom
        candidate[1][0] += beta * (de1 - pred1) * nx / denom
        candidate[1][1] += beta * (de1 - pred1) * ny / denom
        limit = float(CFG["jacobian_max_element_change_ratio"])
        for row in range(2):
            for col in range(2):
                old = self.j[row][col]
                if abs(candidate[row][col] - old) > max(0.02, abs(old) * limit):
                    self.reject_count += 1
                    self.consecutive_rejects += 1
                    return False
                if self.initial is not None:
                    initial_limit = max(0.08, abs(self.initial[row][col]) *
                                        float(CFG["jacobian_max_initial_ratio"]))
                    if abs(candidate[row][col]) > initial_limit:
                        self.reject_count += 1
                        self.consecutive_rejects += 1
                        return False
        if not self.validate(candidate):
            self.reject_count += 1
            self.consecutive_rejects += 1
            return False
        self.j = candidate
        self.update_count += 1
        self.consecutive_rejects = 0
        return True


class CoarseController:
    def __init__(self, jacobian):
        self.jacobian = jacobian
        self.prev_norm = None

    def reset(self):
        self.prev_norm = None

    def compute(self, error, dt):
        norm = math.sqrt(error[0] ** 2 + error[1] ** 2)
        ratio = clamp((norm - float(CFG["coarse_near_px"])) /
                      max(1.0, float(CFG["coarse_far_px"]) - float(CFG["coarse_near_px"])), 0.0, 1.0)
        gain = float(CFG["coarse_gain"]) * (float(CFG["coarse_gain_min_ratio"]) +
               (1.0 - float(CFG["coarse_gain_min_ratio"])) * ratio)
        nx, ny = self.jacobian.solve(error, gain=1.0)
        x_hz, y_hz = gain * nx, gain * ny
        if self.prev_norm is not None and norm < self.prev_norm:
            rate = (norm - self.prev_norm) / max(0.01, dt)
            if norm + rate * 0.08 < float(CFG["fine_enter_px"]):
                x_hz *= 0.35
                y_hz *= 0.35
        self.prev_norm = norm
        return (clamp(x_hz, -float(CFG["x_max_hz"]), float(CFG["x_max_hz"])),
                clamp(y_hz, -float(CFG["y_max_hz"]), float(CFG["y_max_hz"])), nx, ny)


class FinePulseController:
    def __init__(self, jacobian):
        self.jacobian = jacobian
        self.dynamic_x = int(CFG["fine_max_pulses_x"])
        self.dynamic_y = int(CFG["fine_max_pulses_y"])
        self.no_improve = 0

    def reset(self):
        self.dynamic_x = int(CFG["fine_max_pulses_x"])
        self.dynamic_y = int(CFG["fine_max_pulses_y"])
        self.no_improve = 0

    def compute(self, error):
        nx, ny = self.jacobian.solve(error, gain=float(CFG["fine_gain"]), damping=0.001)
        px = int(round(clamp(nx, -self.dynamic_x, self.dynamic_x)))
        py = int(round(clamp(ny, -self.dynamic_y, self.dynamic_y)))
        return px, py

    def record_improvement(self, ratio):
        if ratio <= 0.01:
            self.no_improve += 1
            self.dynamic_x = max(1, self.dynamic_x // 2)
            self.dynamic_y = max(1, self.dynamic_y // 2)
        else:
            self.no_improve = 0


class SafetySupervisor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.fault_code = "NONE"
        self.fault_axis = "-"
        self.total_action = 0.0

    def fault(self, code, axis="-", detail=""):
        self.fault_code = str(code)
        self.fault_axis = str(axis)
        return (self.fault_code, self.fault_axis, str(detail))

    def allow_pulse_action(self, px, py):
        maximum = int(CFG["max_single_action_pulses"])
        if abs(px) > maximum or abs(py) > maximum:
            return False
        self.total_action += abs(px) + abs(py)
        return self.total_action <= float(CFG["max_total_action_pulses"])

    @staticmethod
    def mechanical_limit_active(axis_name, direction):
        # 当前无可用限位GPIO；接口预留，接入开关后只需替换此函数。
        return False


class IBVSSystemController:
    def __init__(self):
        self.camera = CameraManager()
        self.common_enable = CommonEnable()
        self.x_axis = StepperAxis("X", X_STEP_PIN, X_PWM_CHANNEL, X_DIR_PIN, X_POSITIVE_DIR_LEVEL,
                                  X_REVERSE, CFG["x_min_hz"], CFG["x_max_hz"],
                                  CFG["x_accel_hz_s"], CFG["x_decel_hz_s"], X_SOFT_LIMIT_STEPS)
        self.y_axis = StepperAxis("Y", Y_STEP_PIN, Y_PWM_CHANNEL, Y_DIR_PIN, Y_POSITIVE_DIR_LEVEL,
                                  Y_REVERSE, CFG["y_min_hz"], CFG["y_max_hz"],
                                  CFG["y_accel_hz_s"], CFG["y_decel_hz_s"], Y_SOFT_LIMIT_STEPS)
        self.common_enable.enable()
        self.laser_output = LaserTTL35()
        self.laser_output.off()
        self.uart = UARTLink()
        self.target_detector = TargetDetector()
        self.laser_detector = LaserDetector()
        self.legacy_laser = LaserSpotTrackerScaled()
        self.diff_verifier = DifferenceLaserVerifier()
        self.visual = VisualMeasurement()
        self.jacobian = JacobianEstimator()
        self.coarse = CoarseController(self.jacobian)
        self.fine = FinePulseController(self.jacobian)
        self.safety = SafetySupervisor()
        self.state = IBVS_INIT
        self.state_enter_ms = time.ticks_ms()
        self.tracking_enabled = True
        self.target_count = 0
        self.laser_count = 0
        self.target_history = []
        self.laser_history = []
        self.visual_miss = 0
        self.reacquire_start_ms = 0
        self.last_frame_ms = time.ticks_ms()
        self.fps = 0.0
        self.last_plot_ms = 0
        self.last_vision_ms = 0
        self.last_diag_ms = 0
        self.last_status_ms = 0
        self.last_error = None
        self.last_target = None
        self.last_laser = None
        self.last_measurement = None
        self.last_commands = (0.0, 0.0)
        self.lock_errors = []
        self.lock_count = 0
        self.cal_phase = "BEFORE"
        self.cal_samples = []
        self.cal_before = None
        self.cal_pulses = 0
        self.cal_retry = 0
        self.action_start_ms = 0
        self.pending_fine = None
        self.coarse_prev_error = None
        self.coarse_prev_action = None
        self.coarse_prev_saturated = False
        self.calibration_target_center = None
        self.control_ms = self.uart_ms = self.total_ms = 0.0
        self.frame_dt_ms = 0
        self.laser_search_misses = 0
        self.fallback_counter = 0
        self.last_fallback_ms = 0
        self.verify_misses = 0
        self.last_gc_ms = time.ticks_ms()
        self.last_model_action = (0.0, 0.0)
        self.last_improvement = 0.0

    def transition(self, new_state, reason=""):
        if self.state == new_state:
            return
        old = self.state
        self.state = int(new_state)
        self.state_enter_ms = time.ticks_ms()
        if self.state == IBVS_VERIFY_LASER:
            self.verify_misses = 0
        self.uart.send("[state,%d,%d,%s,%s]" % (
            old, self.state, IBVS_STATE_NAME[self.state], str(reason)))
        print("STATE %s -> %s: %s" % (IBVS_STATE_NAME[old], IBVS_STATE_NAME[self.state], reason))

    def stop_motion(self, hard=True):
        if hard:
            self.x_axis.hard_stop()
            self.y_axis.hard_stop()
        else:
            self.x_axis.set_target_hz(0.0)
            self.y_axis.set_target_hz(0.0)
        self.last_commands = (0.0, 0.0)

    def startup_motor_probe(self):
        if not bool(CFG["startup_motor_probe"]):
            return
        pulses = int(CFG["startup_probe_pulses"])
        hz = float(CFG["startup_probe_hz"])
        print("MOTOR_PROBE begin: each axis +%d/-%d pulses at %.0fHz" % (pulses, pulses, hz))
        for axis in (self.x_axis, self.y_axis):
            forward = axis.move_pulses(pulses, hz, 800)
            time.sleep_ms(80)
            backward = axis.move_pulses(-pulses, hz, 800)
            axis.zero_virtual_position()
            print("MOTOR_PROBE %s forward=%s backward=%s net=0" % (
                axis.name, str(forward), str(backward)))
            time.sleep_ms(80)
        print("MOTOR_PROBE done; if the platform did not twitch, check D36A power, EN1/EN2 5V, common GND and STEP wiring")

    def enter_fault(self, code, axis="-", detail=""):
        self.safety.fault(code, axis, detail)
        self.stop_motion(True)
        self.laser_output.off()
        self.transition(IBVS_FAULT, code)
        self.uart.send("[fault,%s,%s,%s]" % (code, axis, str(detail)))

    def restart(self):
        self.stop_motion(True)
        self.laser_output.off()
        self.target_detector.reset()
        self.laser_detector.reset()
        self.legacy_laser.reset()
        self.diff_verifier.active = False
        self.jacobian.reset()
        self.safety.reset()
        self.target_count = self.laser_count = self.visual_miss = 0
        self.verify_misses = 0
        self.last_fallback_ms = 0
        self.target_history = []
        self.laser_history = []
        self.lock_errors = []
        self.lock_count = 0
        self.pending_fine = None
        self.tracking_enabled = True
        self.last_model_action = (0.0, 0.0)
        self.last_improvement = 0.0
        self.transition(IBVS_SEARCH_TARGET, "manual restart")

    def reset_calibration(self):
        self.stop_motion(True)
        self.jacobian.reset()
        self.cal_phase = "BEFORE"
        self.cal_samples = []
        self.cal_before = None
        self.cal_retry = 0
        self.cal_pulses = int(CFG["cal_x_pulses"])
        self.coarse_prev_error = None
        self.coarse_prev_action = None
        self.coarse_prev_saturated = False

    def start_calibration(self):
        self.reset_calibration()
        self.transition(IBVS_CALIBRATE_X, "calibration start")

    def _calibration_step(self, measurement):
        if not measurement.get("valid"):
            return
        if ticks_diff_ms(time.ticks_ms(), self.action_start_ms) < int(CFG["cal_settle_ms"]):
            return
        self.cal_samples.append(measurement["error"])
        if len(self.cal_samples) < int(CFG["cal_samples"]):
            return
        med = median_point(self.cal_samples)
        self.cal_samples = []
        axis_index = 0 if self.state == IBVS_CALIBRATE_X else 1
        axis = self.x_axis if axis_index == 0 else self.y_axis
        if self.cal_phase == "BEFORE":
            self.cal_before = med
            requested = int(self.cal_pulses)
            if not self.safety.allow_pulse_action(requested if axis_index == 0 else 0,
                                                   requested if axis_index == 1 else 0):
                self.enter_fault("CAL_ACTION_LIMIT", axis.name)
                return
            planned, estimated, elapsed = axis.move_pulses(requested, CFG["cal_hz"], CFG["max_axis_run_ms"])
            if planned == 0:
                self.enter_fault("CAL_LIMIT_OR_NO_PULSE", axis.name)
                return
            self.uart.send("[pulse,cal,%s,%d,%d,%d]" % (axis.name, planned, estimated, elapsed))
            self.cal_phase = "AFTER"
            self.action_start_ms = time.ticks_ms()
            return
        de = (med[0] - self.cal_before[0], med[1] - self.cal_before[1])
        shift = math.sqrt(de[0] ** 2 + de[1] ** 2)
        if shift < float(CFG["cal_min_shift_px"]):
            self.cal_retry += 1
            if self.cal_retry >= int(CFG["max_fault_retries"]):
                self.enter_fault("CAL_NO_RESPONSE", axis.name, "shift=%.3f" % shift)
                return
            self.cal_pulses = min(int(CFG["max_single_action_pulses"]), self.cal_pulses * 2)
            self.cal_phase = "BEFORE"
            return
        self.jacobian.set_column(axis_index, de, self.cal_pulses)
        self.cal_phase = "BEFORE"
        self.cal_retry = 0
        if axis_index == 0:
            self.cal_pulses = int(CFG["cal_y_pulses"])
            self.transition(IBVS_CALIBRATE_Y, "X column ready")
        elif not self.jacobian.finalize():
            det, n1, n2, sine = self.jacobian.metrics()
            self.enter_fault("JACOBIAN_INVALID", "XY", "det=%.6f sine=%.3f" % (det, sine))
        else:
            self.calibration_target_center = measurement.get("target")
            self.coarse.reset()
            self.fine.reset()
            norm = math.sqrt(med[0] ** 2 + med[1] ** 2)
            self.transition(IBVS_FINE_AIM if norm <= CFG["fine_enter_px"] else IBVS_COARSE_AIM,
                            "Jacobian valid")

    def _update_coarse(self, measurement, dt):
        if ticks_diff_ms(time.ticks_ms(), self.action_start_ms) < int(CFG["coarse_settle_ms"]):
            return
        error = measurement["error"]
        norm = math.sqrt(error[0] ** 2 + error[1] ** 2)
        if norm <= float(CFG["fine_enter_px"]):
            self.stop_motion(True)
            self.fine.reset()
            self.pending_fine = None
            self.transition(IBVS_FINE_AIM, "fine threshold")
            return
        if (self.coarse_prev_error is not None and self.coarse_prev_action is not None and
                not self.coarse_prev_saturated and CFG["jacobian_update_enable"]):
            self.jacobian.update(self.coarse_prev_error, error, self.coarse_prev_action, 0.10)
            if self.jacobian.consecutive_rejects >= 6:
                self.start_calibration()
                return
        x_request, y_request, nx, ny = self.coarse.compute(error, dt)
        self.last_model_action = (nx, ny)
        px = int(round(clamp(x_request, -float(CFG["coarse_max_pulses_x"]),
                             float(CFG["coarse_max_pulses_x"]))))
        py = int(round(clamp(y_request, -float(CFG["coarse_max_pulses_y"]),
                             float(CFG["coarse_max_pulses_y"]))))
        j = self.jacobian.j
        pred_ex = j[0][0] * px + j[0][1] * py
        pred_ey = j[1][0] * px + j[1][1] * py
        if error[0] * pred_ex + error[1] * pred_ey >= 0.0:
            px = py = 0
        if px == 0 and py == 0:
            self.stop_motion(True)
            return
        if not self.safety.allow_pulse_action(px, py):
            self.enter_fault("COARSE_ACTION_LIMIT", "XY")
            return
        actual_x = actual_y = 0
        # GPIO42/PWM0与GPIO43/PWM1共用频率时钟，因此必须分轴顺序发脉冲。
        if px:
            planned, estimated, elapsed = self.x_axis.move_pulses(
                px, CFG["coarse_pulse_hz"], CFG["max_axis_run_ms"])
            if planned == 0:
                self.enter_fault("SOFT_LIMIT", "X")
                return
            actual_x = planned
            self.uart.send("[pulse,coarse,X,%d,%d,%d]" % (planned, estimated, elapsed))
        if py:
            planned, estimated, elapsed = self.y_axis.move_pulses(
                py, CFG["coarse_pulse_hz"], CFG["max_axis_run_ms"])
            if planned == 0:
                self.enter_fault("SOFT_LIMIT", "Y")
                return
            actual_y = planned
            self.uart.send("[pulse,coarse,Y,%d,%d,%d]" % (planned, estimated, elapsed))
        if self.x_axis.limit_hit or self.y_axis.limit_hit:
            self.enter_fault("SOFT_LIMIT", "X" if self.x_axis.limit_hit else "Y")
            return
        self.last_commands = (actual_x, actual_y)
        self.coarse_prev_error = error
        self.coarse_prev_action = (actual_x, actual_y)
        self.coarse_prev_saturated = (
            abs(actual_x) >= int(CFG["coarse_max_pulses_x"]) or
            abs(actual_y) >= int(CFG["coarse_max_pulses_y"]))
        self.action_start_ms = time.ticks_ms()

    def _update_fine(self, measurement):
        if ticks_diff_ms(time.ticks_ms(), self.action_start_ms) < int(CFG["fine_settle_ms"]):
            return
        error = measurement["error"]
        norm = math.sqrt(error[0] ** 2 + error[1] ** 2)
        if norm > float(CFG["fine_exit_px"]):
            self.pending_fine = None
            self.coarse.reset()
            self.transition(IBVS_COARSE_AIM, "fine hysteresis exit")
            return
        if self.pending_fine is not None:
            before, action, saturated = self.pending_fine
            old_norm = math.sqrt(before[0] ** 2 + before[1] ** 2)
            improvement = (old_norm - norm) / max(0.1, old_norm)
            self.last_improvement = improvement
            self.fine.record_improvement(improvement)
            if CFG["jacobian_update_enable"] and not saturated:
                self.jacobian.update(before, error, action, CFG["jacobian_beta"])
            self.pending_fine = None
            if self.fine.no_improve >= int(CFG["fine_no_improve_limit"]):
                self.start_calibration()
                return
        self.lock_errors.append(error)
        if len(self.lock_errors) > int(CFG["lock_confirm_frames"]):
            self.lock_errors.pop(0)
        within = abs(error[0]) <= CFG["lock_x_px"] and abs(error[1]) <= CFG["lock_y_px"]
        if within and point_variance(self.lock_errors) <= float(CFG["lock_variance_max"]):
            self.lock_count += 1
        else:
            self.lock_count = 0
        if self.lock_count >= int(CFG["lock_confirm_frames"]):
            self.stop_motion(True)
            self.transition(IBVS_LOCKED, "precision confirmed")
            return
        px, py = self.fine.compute(error)
        self.last_model_action = (px, py)
        if px == 0 and py == 0:
            return
        if not self.safety.allow_pulse_action(px, py):
            self.enter_fault("FINE_ACTION_LIMIT", "XY")
            return
        actual_x = actual_y = 0
        saturated = abs(px) >= self.fine.dynamic_x or abs(py) >= self.fine.dynamic_y
        if px:
            planned, estimated, elapsed = self.x_axis.move_pulses(px, CFG["fine_hz"], CFG["max_axis_run_ms"])
            if planned == 0:
                self.enter_fault("SOFT_LIMIT", "X")
                return
            actual_x = planned
            self.uart.send("[pulse,fine,X,%d,%d,%d]" % (planned, estimated, elapsed))
        if py:
            planned, estimated, elapsed = self.y_axis.move_pulses(py, CFG["fine_hz"], CFG["max_axis_run_ms"])
            if planned == 0:
                self.enter_fault("SOFT_LIMIT", "Y")
                return
            actual_y = planned
            self.uart.send("[pulse,fine,Y,%d,%d,%d]" % (planned, estimated, elapsed))
        self.last_commands = (actual_x, actual_y)
        self.pending_fine = (error, (actual_x, actual_y), saturated)
        self.action_start_ms = time.ticks_ms()

    def _update_locked(self, measurement):
        self.stop_motion(True)
        if not measurement.get("valid"):
            return
        ex, ey = measurement["error"]
        if abs(ex) > float(CFG["unlock_x_px"]) or abs(ey) > float(CFG["unlock_y_px"]):
            self.fine.reset()
            self.pending_fine = None
            self.lock_count = 0
            self.transition(IBVS_FINE_AIM, "unlock hysteresis")

    def update_state(self, target, laser, measurement, dt):
        if self.state in (IBVS_FAULT, IBVS_ESTOP):
            self.stop_motion(True)
            return
        if not self.tracking_enabled:
            self.stop_motion(True)
            return
        now_ms = time.ticks_ms()
        state_age_ms = ticks_diff_ms(now_ms, self.state_enter_ms)
        # 无论环境里是否真有靶或激光，状态机都必须可恢复：
        # SEARCH周期重建检测器，其他前置状态超时后明确退回重试。
        if self.state == IBVS_SEARCH_TARGET and state_age_ms > int(CFG["search_rearm_ms"]):
            self.target_detector.reset()
            self.target_count = 0
            self.target_history = []
            self.state_enter_ms = now_ms
            print("WATCHDOG SEARCH_TARGET detector rearmed")
        elif self.state == IBVS_LOCK_TARGET and state_age_ms > int(CFG["lock_target_timeout_ms"]):
            self.target_detector.reset()
            self.transition(IBVS_SEARCH_TARGET, "lock target timeout")
            return
        elif self.state == IBVS_SEARCH_LASER and state_age_ms > int(CFG["search_laser_timeout_ms"]):
            self.diff_verifier.active = False
            self.laser_detector.reset()
            self.laser_output.off()
            self.transition(IBVS_SEARCH_TARGET, "laser search timeout")
            return
        elif self.state == IBVS_VERIFY_LASER and state_age_ms > int(CFG["verify_laser_timeout_ms"]):
            self.laser_history = []
            self.laser_count = 0
            self.transition(IBVS_SEARCH_LASER, "laser verify timeout")
            return
        control_states = (IBVS_CALIBRATE_X, IBVS_CALIBRATE_Y, IBVS_COARSE_AIM,
                          IBVS_FINE_AIM, IBVS_LOCKED)
        if self.state in control_states:
            self.visual_miss = 0 if measurement.get("valid") else self.visual_miss + 1
            if self.visual_miss > int(CFG["max_visual_miss_frames"]):
                self.stop_motion(True)
                self.reacquire_start_ms = time.ticks_ms()
                self.transition(IBVS_REACQUIRE, "visual measurement lost")
                return
        if self.state == IBVS_SEARCH_TARGET:
            self.stop_motion(True)
            self.laser_output.off()
            self.target_count = self.target_count + 1 if target.get("valid") else 0
            if self.target_count >= int(CFG["target_confirm_frames"]):
                self.target_count = 0
                self.target_history = []
                self.transition(IBVS_LOCK_TARGET, "target detected")
        elif self.state == IBVS_LOCK_TARGET:
            self.stop_motion(True)
            if target.get("valid"):
                self.target_history.append(target["center"])
                if len(self.target_history) > int(CFG["target_confirm_frames"]):
                    self.target_history.pop(0)
                stable = (len(self.target_history) >= int(CFG["target_confirm_frames"]) and
                          point_variance(self.target_history) <= float(CFG["target_stability_var"]))
                self.target_count = len(self.target_history) if stable else 0
            else:
                self.target_history = []
                self.target_count = 0
            if self.target_count >= int(CFG["target_confirm_frames"]):
                self.laser_output.on()
                self.laser_count = 0
                self.laser_history = []
                self.transition(IBVS_SEARCH_LASER, "target stable")
        elif self.state == IBVS_SEARCH_LASER:
            self.stop_motion(True)
            if not target.get("valid"):
                self.transition(IBVS_SEARCH_TARGET, "target lost before laser")
                return
            if laser.get("valid"):
                self.laser_count = 1
                self.laser_history = [laser["center"]]
                self.laser_search_misses = 0
                self.transition(IBVS_VERIFY_LASER, "laser candidate")
            else:
                self.laser_search_misses += 1
                if self.laser_search_misses >= 8 and not self.diff_verifier.active:
                    self.diff_verifier.start(self.laser_detector._roi(target.get("outer_box")), self.laser_output)
                    self.laser_search_misses = 0
        elif self.state == IBVS_VERIFY_LASER:
            self.stop_motion(True)
            if measurement.get("valid"):
                self.verify_misses = 0
                self.laser_history.append(measurement["laser"])
                if len(self.laser_history) > int(CFG["laser_confirm_frames"]):
                    self.laser_history.pop(0)
                stable = (len(self.laser_history) >= int(CFG["laser_confirm_frames"]) and
                          point_variance(self.laser_history) <= float(CFG["laser_stability_var"]))
                self.laser_count = len(self.laser_history) if stable else 0
            else:
                self.verify_misses += 1
                if self.verify_misses > int(CFG["verify_miss_tolerance"]):
                    self.laser_history = []
                    self.laser_count = 0
            if (self.laser_count >= int(CFG["laser_confirm_frames"])
                    and measurement.get("valid")
                    and not measurement.get("target_predicted")):
                self.start_calibration()
        elif self.state in (IBVS_CALIBRATE_X, IBVS_CALIBRATE_Y):
            self._calibration_step(measurement)
        elif self.state == IBVS_COARSE_AIM and measurement.get("valid"):
            self._update_coarse(measurement, dt)
        elif self.state == IBVS_FINE_AIM and measurement.get("valid"):
            self._update_fine(measurement)
        elif self.state == IBVS_LOCKED:
            self._update_locked(measurement)
        elif self.state == IBVS_REACQUIRE:
            self.stop_motion(True)
            if measurement.get("valid"):
                self.laser_count += 1
                if self.laser_count >= int(CFG["laser_confirm_frames"]):
                    self.laser_count = 0
                    norm = math.sqrt(measurement["error"][0] ** 2 + measurement["error"][1] ** 2)
                    moved = False
                    if self.calibration_target_center is not None:
                        dx = measurement["target"][0] - self.calibration_target_center[0]
                        dy = measurement["target"][1] - self.calibration_target_center[1]
                        moved = dx * dx + dy * dy > float(CFG["recalibrate_target_shift_px"]) ** 2
                    if self.jacobian.valid and not moved:
                        self.transition(IBVS_FINE_AIM if norm <= CFG["fine_enter_px"] else IBVS_COARSE_AIM,
                                        "measurement reacquired")
                    else:
                        self.start_calibration()
            else:
                self.laser_count = 0
            if ticks_diff_ms(time.ticks_ms(), self.reacquire_start_ms) > int(CFG["max_reacquire_ms"]):
                self.laser_output.off()
                self.laser_detector.reset()
                self.target_detector.reset()
                self.transition(IBVS_SEARCH_TARGET, "reacquire timeout")

    def apply_slider(self, name, raw):
        try:
            value = float(raw)
        except Exception:
            self.uart.send("[ack,error,bad_value,%s]" % name)
            return
        key = str(name).strip().lower()
        mapping = {
            "coarsegain": ("coarse_gain", 0.05, 3.0),
            "finegain": ("fine_gain", 0.05, 2.0),
            "fineenterpx": ("fine_enter_px", 2.0, 40.0),
            "fineexitpx": ("fine_exit_px", 3.0, 60.0),
            "lockxpx": ("lock_x_px", 0.3, 8.0), "lockypx": ("lock_y_px", 0.3, 8.0),
            "xmaxhz": ("x_max_hz", 50.0, 1200.0), "ymaxhz": ("y_max_hz", 50.0, 1000.0),
            "xaccel": ("x_accel_hz_s", 200.0, 10000.0), "yaccel": ("y_accel_hz_s", 200.0, 10000.0),
            "calxpulses": ("cal_x_pulses", 4.0, 100.0), "calypulses": ("cal_y_pulses", 4.0, 100.0),
            "finemaxpulsesx": ("fine_max_pulses_x", 1.0, 100.0),
            "finemaxpulsesy": ("fine_max_pulses_y", 1.0, 100.0),
            "jacobianbeta": ("jacobian_beta", 0.01, 0.8),
        }
        if key not in mapping:
            self.uart.send("[ack,error,unknown_slider,%s]" % name)
            return
        cfg_key, low, high = mapping[key]
        value = clamp(value, low, high)
        if "pulses" in cfg_key:
            value = int(round(value))
        CFG[cfg_key] = value
        if cfg_key == "x_max_hz": self.x_axis.max_hz = float(value)
        if cfg_key == "y_max_hz": self.y_axis.max_hz = float(value)
        if cfg_key == "x_accel_hz_s": self.x_axis.accel_hz_s = float(value)
        if cfg_key == "y_accel_hz_s": self.y_axis.accel_hz_s = float(value)
        self.uart.send("[ack,slider,%s,%s]" % (name, str(value)))

    def _jog(self, axis, hz, duration_ms):
        self.stop_motion(True)
        start = time.ticks_ms()
        last = start
        axis.set_target_hz(hz)
        while ticks_diff_ms(time.ticks_ms(), start) < duration_ms:
            now = time.ticks_ms()
            axis.update(max(0.001, ticks_diff_ms(now, last) / 1000.0))
            last = now
            time.sleep_ms(5)
        axis.hard_stop()

    def handle_uart(self):
        for parts in self.uart.read_packets(UART_MAX_PACKETS_PER_LOOP):
            if not parts:
                continue
            typ = parts[0].strip().lower()
            cmd = parts[1].strip().lower() if len(parts) > 1 else ""
            if typ == "slider" and len(parts) >= 3:
                self.apply_slider(parts[1], parts[2])
            elif typ in ("tracking", "key") and cmd in ("start", "mode"):
                self.tracking_enabled = True
                if self.state in (IBVS_FAULT, IBVS_ESTOP): self.restart()
            elif typ in ("tracking", "key") and cmd == "stop":
                self.tracking_enabled = False
                self.stop_motion(True)
            elif typ == "laser" and cmd == "on":
                if self.state in (IBVS_FAULT, IBVS_ESTOP):
                    self.uart.send("[ack,error,safety_state_laser_blocked]")
                else:
                    self.laser_output.on()
            elif typ == "laser" and cmd == "off":
                self.laser_output.off(); self.stop_motion(True)
            elif typ == "calibrate" and cmd == "start":
                if self.state in (IBVS_FAULT, IBVS_ESTOP):
                    self.uart.send("[ack,error,safety_state_calibration_blocked]")
                else:
                    self.start_calibration()
            elif typ == "calibrate" and cmd == "reset":
                self.stop_motion(True); self.jacobian.reset(); self.transition(IBVS_VERIFY_LASER, "calibration reset")
            elif typ == "jacobian" and cmd == "print": self.send_jacobian()
            elif typ in ("estop", "sv") and (typ == "estop" or cmd == "estop"):
                self.stop_motion(True); self.laser_output.off(); self.transition(IBVS_ESTOP, "UART estop")
            elif typ in ("restart", "sv") and (typ == "restart" or cmd in ("restart", "resume")):
                self.restart()
            elif typ == "motor" and cmd == "test":
                if self.state in (IBVS_FAULT, IBVS_ESTOP):
                    self.uart.send("[ack,error,safety_state_motor_blocked]")
                    continue
                self.stop_motion(True)
                for axis in (self.x_axis, self.y_axis):
                    axis.move_pulses(16, 120, 500); time.sleep_ms(100)
                    axis.move_pulses(-16, 120, 500); time.sleep_ms(100)
                self.jacobian.reset(); self.target_detector.reset(); self.laser_detector.reset()
                self.transition(IBVS_SEARCH_TARGET, "motor test invalidated calibration")
            elif typ == "motor" and cmd == "jog" and len(parts) >= 5:
                if self.state in (IBVS_FAULT, IBVS_ESTOP):
                    self.uart.send("[ack,error,safety_state_motor_blocked]")
                    continue
                axis = self.x_axis if parts[2].strip().lower() == "x" else self.y_axis
                self._jog(axis, clamp(float(parts[3]), -450, 450), int(clamp(float(parts[4]), 50, 1500)))
                self.jacobian.reset(); self.target_detector.reset(); self.laser_detector.reset()
                self.transition(IBVS_SEARCH_TARGET, "manual jog invalidated calibration")
            elif typ == "system" and cmd == "ping": self.uart.send("[system,pong,v9_4_1]")

    def send_jacobian(self):
        det, _, _, _ = self.jacobian.metrics()
        self.uart.send("[jacobian,%.7f,%.7f,%.7f,%.7f,%.9f,%d,%d,%d]" % (
            self.jacobian.j[0][0], self.jacobian.j[0][1], self.jacobian.j[1][0],
            self.jacobian.j[1][1], det, 1 if self.jacobian.valid else 0,
            self.jacobian.update_count, self.jacobian.reject_count))

    def send_telemetry(self, target, laser, measurement):
        now = time.ticks_ms()
        if measurement.get("valid"):
            ex, ey = measurement["error"]
            tx, ty = measurement["target"]
            lx, ly = measurement["laser"]
            norm = math.sqrt(ex * ex + ey * ey)
        else:
            ex = ey = norm = 0.0
            tx = ty = lx = ly = -1.0
        if ticks_diff_ms(now, self.last_plot_ms) >= int(CFG["plot_interval_ms"]):
            self.last_plot_ms = now
            self.uart.send("[plot,%.3f,%.3f,%.3f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%d]" % (
                ex, ey, norm, self.last_commands[0], self.last_commands[1], lx, ly, tx, ty, self.state))
        if ticks_diff_ms(now, self.last_vision_ms) >= int(CFG["vision_interval_ms"]):
            self.last_vision_ms = now
            self.uart.send("[vision,%d,%.3f,%d,%.3f,%d,%d,%d,%.2f,%.2f,%.2f]" % (
                1 if target.get("valid") else 0, float(target.get("confidence", 0.0)),
                1 if laser.get("valid") else 0, float(laser.get("confidence", 0.0)),
                int(target.get("candidate_count", 0)), int(laser.get("candidate_count", 0)),
                self.frame_dt_ms, float(target.get("detect_ms", 0.0)),
                float(laser.get("detect_ms", 0.0)), self.fps))
        if ticks_diff_ms(now, self.last_diag_ms) >= int(CFG["diag_interval_ms"]):
            self.last_diag_ms = now
            self.send_jacobian()
            self.uart.send("[control,%d,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.3f,%d]" % (
                self.state, ex, ey, self.last_model_action[0], self.last_model_action[1],
                self.x_axis.applied_hz, self.y_axis.applied_hz, self.last_improvement, self.lock_count))
            self.uart.send("[perf,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d]" % (
                self.camera.capture_ms, float(target.get("detect_ms", 0.0)),
                float(laser.get("detect_ms", 0.0)), self.control_ms,
                self.uart_ms, self.total_ms, self.frame_dt_ms))
            print("STAT %s valid=%d T=%d/%.2f/b%.2f/n%d/%s/%s L=%d/%.2f/n%d/%s e=(%.2f,%.2f) cmd=(%.1f,%.1f) J=%d fps=%.1f ms=%d/%.1f/%.1f/%.1f" % (
                IBVS_STATE_NAME[self.state], 1 if measurement.get("valid") else 0,
                1 if target.get("valid") else 0, float(target.get("confidence", 0.0)),
                float(target.get("best_quality", 0.0)), int(target.get("candidate_count", 0)),
                str(target.get("source", "NONE")), str(target.get("reject_reason", "-")),
                1 if laser.get("valid") else 0, float(laser.get("confidence", 0.0)),
                int(laser.get("candidate_count", 0)), str(laser.get("source", "NONE")),
                ex, ey, self.last_commands[0], self.last_commands[1],
                1 if self.jacobian.valid else 0, self.fps, self.frame_dt_ms,
                self.camera.capture_ms, float(target.get("detect_ms", 0.0)),
                float(laser.get("detect_ms", 0.0))))

    def run(self):
        try:
            print("=" * 78)
            print("K230 D36A LASER TARGET IBVS V9.4.1 PWM API COMPAT")
            print("STEP/DIR X=GPIO42/26 Y=GPIO43/34 UART3=GPIO32/33 LASER=GPIO35")
            print("Control coordinates: 256x192 detect pixels; software safety range: +/-%.0f deg" % SOFT_LIMIT_DEG)
            print("States: INIT SEARCH_TARGET LOCK_TARGET SEARCH_LASER VERIFY_LASER CAL_X CAL_Y COARSE FINE LOCKED REACQUIRE FAULT ESTOP")
            print("Commands: tracking start/stop; calibrate start/reset; jacobian print; estop; restart; motor test/jog")
            print("=" * 78)
            self.startup_motor_probe()
            self.camera.start()
            self.transition(IBVS_SEARCH_TARGET, "camera ready")
            self.last_frame_ms = time.ticks_ms()
            self.fps = 0.0
            self.uart.send("[system,ready,k230_d36a_laser_target_ibvs_v9_4_1_pwm_api_compat]")
            while True:
                os.exitpoint()
                loop_start = time.ticks_us()
                now_ms = time.ticks_ms()
                self.frame_dt_ms = max(1, ticks_diff_ms(now_ms, self.last_frame_ms))
                self.last_frame_ms = now_ms
                inst_fps = 1000.0 / self.frame_dt_ms
                self.fps = inst_fps if self.fps <= 0 else 0.86 * self.fps + 0.14 * inst_fps
                dt = clamp(self.frame_dt_ms / 1000.0, 0.005, 0.20)
                img, img_np, timestamp_us = self.camera.capture()
                allow_target_prediction = self.state in (
                    IBVS_LOCK_TARGET, IBVS_SEARCH_LASER, IBVS_VERIFY_LASER, IBVS_REACQUIRE)
                target = self.target_detector.detect(
                    img, img_np, dt, timestamp_us, allow_target_prediction)
                if self.laser_output.enabled:
                    laser = self.laser_detector.detect(
                        img, img_np, target.get("outer_box"), timestamp_us)
                else:
                    laser = {"valid": False, "center": None, "confidence": 0.0,
                             "candidate_count": 0, "ambiguous": False, "source": "OFF",
                             "timestamp_us": timestamp_us, "detect_ms": 0.0}
                # 红色指数首次捕获失败时，低频调用已验证的RGB565 CORE/HALO链路。
                # 其时间戳不伪装成同帧；超出12ms只用于建立锚点，不直接闭环。
                if (self.laser_output.enabled and not laser.get("valid") and
                        self.state == IBVS_SEARCH_LASER and
                        not self.diff_verifier.active):
                    now_fallback_ms = time.ticks_ms()
                    if ticks_diff_ms(now_fallback_ms, self.last_fallback_ms) >= int(CFG["fallback_min_interval_ms"]):
                        self.last_fallback_ms = now_fallback_ms
                        fb_img, fb_np, fb_ts = self.camera.capture_laser_fallback()
                        desired = target.get("center")
                        desired_logical = None if desired is None else (
                            detect_to_logical_x(desired[0]), detect_to_logical_y(desired[1]))
                        fb_pos = self.legacy_laser.detect(
                            fb_img, fb_np, target.get("outer_box"), desired_logical, dt)
                        if fb_pos is not None and self.legacy_laser.locked:
                            fb_result = {
                                "valid": True,
                                "center": (logical_to_detect_x(fb_pos[0]), logical_to_detect_y(fb_pos[1])),
                                "area": float(self.legacy_laser.last_area), "peak": 0.0,
                                "confidence": float(self.legacy_laser.confidence),
                                "candidate_count": int(self.legacy_laser.candidate_count),
                                "ambiguous": bool(self.legacy_laser.ambiguous),
                                "source": str(self.legacy_laser.last_source),
                                "timestamp_us": int(fb_ts),
                                "detect_ms": float(self.camera.fallback_capture_ms),
                            }
                            self.laser_detector.accept_external(fb_result)
                            laser = fb_result
                if self.diff_verifier.active:
                    diff = self.diff_verifier.update(img_np, timestamp_us, self.laser_output)
                    if diff is not None and diff.get("valid"):
                        laser = diff
                        self.laser_detector.accept_external(diff)
                measurement = self.visual.build(target, laser)
                control_start = time.ticks_us()
                self.update_state(target, laser, measurement, dt)
                self.control_ms = perf_ms(control_start)
                uart_start = time.ticks_us()
                self.handle_uart()
                self.send_telemetry(target, laser, measurement)
                self.uart_ms = perf_ms(uart_start)
                self.total_ms = perf_ms(loop_start)
                self.last_target, self.last_laser, self.last_measurement = target, laser, measurement
                if ticks_diff_ms(now_ms, self.last_gc_ms) >= 3000 and gc.mem_free() < GC_FREE_THRESHOLD:
                    self.last_gc_ms = now_ms
                    gc.collect()
                time.sleep_ms(1)
        except KeyboardInterrupt:
            print("IDE interrupt")
        except BaseException as e:
            # CanMV IDE的停止按钮在某些底层调用中抛出的不是
            # KeyboardInterrupt，但它仍是用户主动停止，不应记为系统FAULT。
            if "IDE interrupt" in str(e):
                print("IDE interrupt: safe stop")
            else:
                print("FATAL", e)
                try:
                    sys.print_exception(e)
                except Exception:
                    pass
                self.enter_fault("UNHANDLED_EXCEPTION", "-", str(e))
        finally:
            self.stop_motion(True)
            self.laser_output.off()
            self.camera.close()
            try:
                MediaManager.deinit()
            except Exception:
                pass
            self.uart.close()
            self.x_axis.deinit()
            self.y_axis.deinit()
            self.common_enable.deinit()
            self.laser_output.deinit()
            print("program exited; X/Y STEP stopped; laser OFF; UART/camera/media released")


if __name__ == "__main__":
    IBVSSystemController().run()
