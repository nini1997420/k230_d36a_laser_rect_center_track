# -*- coding: utf-8 -*-
"""
K230 + D36A 固定跑道静止靶持续锁定 v8.1

运行环境：CanMV IDE K230 / MicroPython
视觉基准：main_k230_rect_track_uart_v13_speed_recovery.py
执行器：D36A 双路步进驱动板 + 两相步进电机

功能：
- 256x192 RGB888采集，控制和串口仍使用640x480逻辑坐标；
- cv_lite主检测 + 可自动降级的原生find_rects + Kalman动态ROI + 两级关联；
- 外框/内框嵌套评分、黑框白心对比度、可选亚像素角点精修；
- SEARCH / ACQUIRE / TRACK / COAST / LOST 状态机；
- K230 两路硬件 PWM 直接产生 STEP 脉冲；
- GPIO 控制 X/Y 方向；
- 静止靶三段PD精瞄 + 低冲击Ramp + 高频无响应自动降速重试；
- 可信首帧立即出步；目标身份保留5秒，电机仅桥接1~2个漏检帧；
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
GPIO35用于激光TTL控制。严禁把 D36A 的 5V 接到任何 K230 GPIO。

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
- 软件角度限位已完全关闭；机械限位、绕线与碰撞风险必须由现场保证。
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
SOFT_LIMITS_ENABLED = False
X_SOFT_LIMIT_STEPS = 0.0
Y_SOFT_LIMIT_STEPS = 0.0

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

# ============================================================
# 4. 步进速度闭环初始参数
# ============================================================
CONTROL_DT_MIN = 0.012
CONTROL_DT_MAX = 0.180
ZERO_CROSS_BRAKE_PX = 12.0
FAST_VEL_START_PX_S = 45.0
FAST_ERR_START_PX = 12.0

PID_CFG = {
    # 输出单位为 STEP Hz，不再是舵机角度增量。
    "x_kp": 8.20,
    "x_ki": 0.000,
    "x_kd": 0.200,
    "x_deadzone_px": 0.8,

    "y_kp": 6.80,
    "y_ki": 0.000,
    "y_kd": 0.200,
    "y_deadzone_px": 1.0,

    "integral_limit": 180.0,
    "derivative_tau_s": 0.070,
}

MOTION_CFG = {
    "x_min_hz": 12.0,
    "y_min_hz": 10.0,
    # 数据中X轴在1500Hz持续约17秒却没有可见位移。开环步进提高频率只会
    # 降低可用扭矩，因此先把工作区限制在更可靠的频率，再由响应监视器重试。
    "x_max_hz": 1000.0,
    "y_max_hz": 850.0,

    # 加速保留跟踪能力，减速高于加速，避免高速接近中心后刹车不足。
    "x_accel_hz_s": 3600.0,
    "y_accel_hz_s": 3200.0,
    "x_decel_hz_s": 6200.0,
    "y_decel_hz_s": 5600.0,

    # px/s -> Hz
    "x_ff": 1.00,
    "y_ff": 0.82,
    "x_ff_limit_hz": 900.0,
    "y_ff_limit_hz": 800.0,
    "speed_scale": 1.00,

    # 自适应速度上限：远处允许快，接近中心自动限速，防止高速穿越中心。
    "near_error_px": 18.0,
    "mid_error_px": 55.0,
    "far_error_px": 140.0,
    "x_near_cap_hz": 260.0,
    "y_near_cap_hz": 220.0,
    "x_mid_cap_hz": 900.0,
    "y_mid_cap_hz": 760.0,
    # 与当前可靠工作频率一致，避免串口旧参数绕过新上限。
    "x_far_cap_hz": 1000.0,
    "y_far_cap_hz": 850.0,

    # 目标已经向中心高速接近时，按预计到达时间提前制动。
    "approach_brake_time_s": 0.180,
    "approach_predict_s": 0.045,
    "approach_min_scale": 0.12,
    # 预测将穿过中心时不再把STEP目标直接切为0，而是连续降速。
    "approach_cross_scale": 0.20,
    "zero_cross_scale": 0.35,
    "approach_max_error_px": 68.0,
    "error_floor_start_px": 24.0,

    # 短时漏检使用 Kalman 预测位置重新计算控制。
    # 普通/向中心运动：保守减速，避免穿越中心后继续冲。
    "coast_enable": True,
    "coast_scale_1": 0.82,
    "coast_scale_2": 0.52,
    "coast_scale_3": 0.24,
    "coast_scale_4": 0.08,
    "coast_scale_5": 0.00,
    "coast_scale_6": 0.00,
    "coast_ff_scale": 0.35,
    "coast_predict_stop_s": 0.080,
    "x_coast_max_hz": 320.0,
    "y_coast_max_hz": 260.0,

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
LASER_AUTO_ON_AFTER_CAMERA = True
LASER_PACKET_INTERVAL_MS = 250

# 激光闭环方向与“摄像头中心追矩形”方向不是同一物理映射。
# 某轴越追越远时，只翻转对应一项，或串口发送：
# [slider,laserXReverse,1] / [slider,laserYReverse,1]
LASER_X_REVERSE = False
LASER_Y_REVERSE = False

# 矩形与激光连续稳定多少帧后，才允许 STEP 输出。
LASER_CONTROL_ARM_FRAMES = 1
# 丢失任一测量立即停步进，不做盲目回中。
LASER_COAST_ENABLE = True
# v4.4：修复COAST控制被状态判断再次拦截的问题，并允许短时连续控制。
LASER_PREDICT_MAX_FRAMES = 4
LASER_PREDICT_VEL_DECAY = 0.76
LASER_COAST_CMD_SCALE_1 = 0.88
LASER_COAST_CMD_SCALE_2 = 0.68
RECT_COAST_CMD_SCALE_1 = 0.88
# 误差滤波按距离分区：中心附近抑制像素噪声，远区保留响应速度。
CONTROL_ERROR_ALPHA_NEAR = 0.42
CONTROL_ERROR_ALPHA_MID = 0.60
CONTROL_ERROR_ALPHA_FAR = 0.80
CONTROL_ERROR_ALPHA_JUMP = 0.66
CONTROL_REL_VEL_ALPHA = 0.18
RECT_COAST_CMD_SCALE_2 = 0.72
RECT_COAST_CMD_SCALE_3 = 0.56
RECT_COAST_CMD_SCALE_4 = 0.40
RECT_COAST_CMD_SCALE_5 = 0.27
RECT_COAST_CMD_SCALE_6 = 0.16
RECT_COAST_CMD_SCALE_7 = 0.08
RECT_COAST_CMD_SCALE_8 = 0.04
RECT_COAST_CMD_SCALE_9 = 0.02
RECT_COAST_CMD_SCALE_10 = 0.00
RECT_COAST_MAX_FRAMES = 10
INVALID_HARD_RESET_FRAMES = 4
INVALID_COMMAND_DECAY = 0.62

# v4.5 精定位参数。坐标单位均为640x480逻辑像素。
# 进入停止区后，必须超过退出阈值才重新运动，避免中心附近反复启停。
PRECISION_CFG = {
    "x_stop_enter_px": 3.5,
    "x_stop_exit_px": 6.0,
    "y_stop_enter_px": 3.5,
    "y_stop_exit_px": 6.0,
    # 必须连续多帧处于停止区才关断PWM，避免单帧噪声造成启停。
    "stop_confirm_frames": 3,

    "precision_error_px": 16.0,
    "mid_error_px": 52.0,

    # 小误差区降低P、前馈和速度上限；大误差区保留追踪速度。
    "near_kp_scale": 0.58,
    "near_kd_scale": 0.40,
    "near_ff_scale": 0.00,
    "mid_kp_scale": 0.95,
    "mid_kd_scale": 0.75,
    "mid_ff_scale": 0.55,
    "far_kp_scale": 1.15,
    "far_kd_scale": 0.85,
    "far_ff_scale": 0.95,

    "x_precision_cap_hz": 85.0,
    "y_precision_cap_hz": 70.0,
    "x_mid_cap_hz": 650.0,
    "y_mid_cap_hz": 560.0,
}

# v7.2：依据实测输出阶跃设置分区指令斜率。
# dt 最大只按 50ms 计算，避免偶发慢帧把一次允许变化量放大数倍。
# 近区优先平滑，中区限制突跳，远区仍保留快速追赶；反向必须先过零。
COMMAND_SLEW_CFG = {
    "dt_max_s": 0.050,
    "x_rise_near": 1800.0,
    "x_rise_mid": 4400.0,
    "x_rise_far": 8000.0,
    "x_fall_near": 3000.0,
    "x_fall_mid": 6000.0,
    "x_fall_far": 8500.0,
    "x_reverse": 7000.0,
    "y_rise_near": 1600.0,
    "y_rise_mid": 4000.0,
    "y_rise_far": 7000.0,
    "y_fall_near": 2800.0,
    "y_fall_mid": 5500.0,
    "y_fall_far": 8000.0,
    "y_reverse": 6500.0,
}

# v4.7：视觉边界保护和软限位自动重基准。
# 软件步数没有编码器反馈，长期积分会误触发限位；稳定对准后重新以当前位置为0。
# 用户要求限位严格相对本次启动位置；禁止稳定对准后把当前位置重新当作零点。
AUTO_REBASE_ENABLE = False
AUTO_REBASE_ERR_X_PX = 4.0
AUTO_REBASE_ERR_Y_PX = 4.0
AUTO_REBASE_STABLE_FRAMES = 24
LASER_EDGE_GUARD_X_PX = 10.0
LASER_EDGE_GUARD_Y_PX = 10.0
LASER_PREDICT_X_CAP_HZ = 190.0
LASER_PREDICT_Y_CAP_HZ = 155.0

# 连续发散保护：目标矩形基本静止、已经输出步进命令，误差却持续增大时停机。
DIVERGENCE_GUARD_ENABLE = True
DIVERGENCE_GROW_PX = 4.0
DIVERGENCE_GROW_RATIO = 1.06
DIVERGENCE_LIMIT_FRAMES = 4
DIVERGENCE_RECT_SPEED_MAX = 80.0
# 发散保护只做短暂制动和控制器重置，不再形成必须人工清除的永久锁死。
DIVERGENCE_RECOVERY_FRAMES = 6

# ============================================================
# 固定跑道静止靶控制器
# ============================================================
# 设计原则：检测负责“目标在哪里”，控制只根据当前矩形-激光误差对准。
# 不使用目标速度前馈、不使用Kalman前瞻控制、不在长时间漏检时盲追。
STATIC_AIM_CFG = {
    "fine_error_px": 12.0,
    "mid_error_px": 55.0,
    "x_lock_enter_px": 2.2,
    "x_lock_exit_px": 4.2,
    "y_lock_enter_px": 2.4,
    "y_lock_exit_px": 4.4,
    "lock_confirm_frames": 3,

    # 误差滤波：远区优先响应，精瞄区优先稳定。
    "alpha_far": 0.90,
    "alpha_mid": 0.68,
    "alpha_fine": 0.32,
    "derivative_alpha": 0.24,

    # 三段PD，输出单位为STEP Hz。D项只用于接近中心时提前减速。
    "x_kp_far": 6.4,
    "x_kd_far": 0.10,
    "x_cap_far_hz": 900.0,
    "x_kp_mid": 5.2,
    "x_kd_mid": 0.18,
    "x_cap_mid_hz": 420.0,
    "x_kp_fine": 3.8,
    "x_kd_fine": 0.22,
    "x_cap_fine_hz": 80.0,

    "y_kp_far": 5.6,
    "y_kd_far": 0.10,
    "y_cap_far_hz": 750.0,
    "y_kp_mid": 4.7,
    "y_kd_mid": 0.18,
    "y_cap_mid_hz": 360.0,
    "y_kp_fine": 3.6,
    "y_kd_fine": 0.22,
    "y_cap_fine_hz": 70.0,

    # 仅桥接两个检测漏帧；静止靶不允许用预测位置长时间继续追。
    "miss_hold_frames": 2,
    # 单帧漏检不改变指令，第二帧才降速；避免TRACK/COAST交替造成一顿一顿。
    "miss_scale_1": 1.00,
    "miss_scale_2": 0.65,
}

# 开环步进没有编码器，PWM已加载不代表转轴真实移动。这里用视觉误差作为
# 低带宽响应反馈：连续高频却没有改善时，临时回到高扭矩低速，随后自动重试。
RESPONSE_GUARD_CFG = {
    "min_error_px": 24.0,
    "min_command_hz": 430.0,
    "observe_ms": 750,
    "fast_fail_ms": 360,
    "growth_fail_px": 12.0,
    "min_improve_px": 3.0,
    "min_improve_ratio": 0.04,
    "recovery_ms": 1400,
    "x_recovery_cap_hz": 300.0,
    "y_recovery_cap_hz": 260.0,
}

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
# 单控制器优化版
# ============================================================
class K230D36AStaticLaneTargetingV81:
    """v8.1: 固定跑道静止靶持续锁定与步进响应保护控制器。

由 v4.7 功能等价扁平化生成：保留矩形检测、激光检测、PID、STEP/DIR、UART、
状态机、软限位、诊断和异常清理，删除完整基础应用控制器与激光子类之间的重复实现。"""

    def __init__(self):
        self.cx0 = CAMERA_WIDTH // 2
        self.cy0 = CAMERA_HEIGHT // 2
        self.common_enable = CommonEnable()
        self.x_axis = StepperAxis('X', X_STEP_PIN, X_DIR_PIN, X_POSITIVE_DIR_LEVEL, X_REVERSE, MOTION_CFG['x_min_hz'], MOTION_CFG['x_max_hz'], MOTION_CFG['x_accel_hz_s'], MOTION_CFG['x_decel_hz_s'], X_SOFT_LIMIT_STEPS)
        self.y_axis = StepperAxis('Y', Y_STEP_PIN, Y_DIR_PIN, Y_POSITIVE_DIR_LEVEL, Y_REVERSE, MOTION_CFG['y_min_hz'], MOTION_CFG['y_max_hz'], MOTION_CFG['y_accel_hz_s'], MOTION_CFG['y_decel_hz_s'], Y_SOFT_LIMIT_STEPS)
        self.common_enable.enable()
        print('D36A EN1/EN2 tied to board 5V; GPIO35 controls laser TTL')
        self.x_pid = VelocityPID(PID_CFG['x_kp'], PID_CFG['x_ki'], PID_CFG['x_kd'], MOTION_CFG['x_max_hz'], PID_CFG['integral_limit'], PID_CFG['derivative_tau_s'])
        self.y_pid = VelocityPID(PID_CFG['y_kp'], PID_CFG['y_ki'], PID_CFG['y_kd'], MOTION_CFG['y_max_hz'], PID_CFG['integral_limit'], PID_CFG['derivative_tau_s'])
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
        self.laser_output = LaserTTL35()
        self.laser = LaserSpotTrackerScaled()
        self.current_laser_pos = None
        self.control_arm_count = 0
        self.direction_fault = False
        self.direction_fault_reported = False
        self.divergence_recovery_frames = 0
        self.previous_error_norm = None
        self.divergence_count = 0
        self.last_laser_packet_ms = time.ticks_ms()
        self.filtered_err_x = None
        self.filtered_err_y = None
        self.filtered_rel_vx = 0.0
        self.filtered_rel_vy = 0.0
        self.invalid_control_frames = 0
        self.x_precision_hold = False
        self.y_precision_hold = False
        self.x_stop_confirm_count = 0
        self.y_stop_confirm_count = 0
        self.last_x_zone = 0
        self.last_y_zone = 0
        self.stable_align_count = 0
        self.frame_error_count = 0
        self.last_frame_error_ms = 0
        self.aim_mode = 'WAIT'
        self.static_miss_frames = 0
        self.x_error_derivative = 0.0
        self.y_error_derivative = 0.0
        self.x_aim_locked = False
        self.y_aim_locked = False
        self.x_aim_lock_count = 0
        self.y_aim_lock_count = 0
        self._reset_response_guard()

    @staticmethod
    def crossed_zero(previous, current):
        if previous is None:
            return False
        return previous > 0.0 and current < 0.0 or (previous < 0.0 and current > 0.0)

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

    def decelerate_motion(self, dt):
        """正常丢测量时沿硬件加减速坡道降到0，避免每次状态切换都砍断脉冲。"""
        dt = clamp(safe_float(dt, 0.033), CONTROL_DT_MIN, CONTROL_DT_MAX)
        self.last_x_target_hz = 0.0
        self.last_y_target_hz = 0.0
        self.last_track_x_hz = 0.0
        self.last_track_y_hz = 0.0
        self.x_axis.set_target_hz(0.0)
        self.y_axis.set_target_hz(0.0)
        self.x_axis.update(dt)
        self.y_axis.update(dt)
        self.reset_pid_memory()

    def stop_and_reset_tracker(self, zero_virtual=False):
        self.stop_motion(hard=True)
        self.tracker.reset()
        if zero_virtual:
            self.x_axis.zero_virtual_position()
            self.y_axis.zero_virtual_position()
        print('stepper stopped; tracker reset; zero_virtual=%s' % zero_virtual)

    def emergency_stop(self):
        self.estop = True
        self.stop_motion(hard=True)
        self.common_enable.disable()

    def restart_after_estop(self):
        self.common_enable.enable()
        self.estop = False
        self.stop_motion(hard=True)

    def _pause_motion_preserve_state(self):
        self.last_x_target_hz = 0.0
        self.last_y_target_hz = 0.0
        self.x_axis.hard_stop()
        self.y_axis.hard_stop()

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

    def handle_uart(self):
        packets = self.uart.read_packets(UART_MAX_PACKETS_PER_LOOP)
        latest_slider = {}
        commands = []
        for parts in packets:
            if parts and parts[0].strip().lower() == 'slider' and (len(parts) >= 3):
                latest_slider[parts[1].strip().lower()] = parts[2]
            else:
                commands.append(parts)
        for name, value in latest_slider.items():
            self.apply_slider(name, value)
        for parts in commands:
            if not parts:
                continue
            typ = parts[0].strip().lower()
            cmd = parts[1].strip().lower() if len(parts) >= 2 else ''
            if typ in ('sv', 'key', 'stepper', 'motor') and cmd in ('estop', 'emergency'):
                self.emergency_stop()
                self.uart.send('[stepper,estop,1]')
            elif typ in ('sv', 'stepper', 'motor') and cmd in ('restart', 'resume'):
                self.restart_after_estop()
                self.uart.send('[stepper,estop,0]')
            elif typ == 'key' and cmd in ('start', 'mode'):
                self.tracking_enabled = True
                self.restart_after_estop()
                self.uart.send('[stepper,tracking,1]')
            elif typ == 'key' and cmd == 'stop':
                self.tracking_enabled = False
                self.stop_motion(hard=True)
                self.uart.send('[stepper,tracking,0]')
            elif typ in ('stepper', 'motor') and cmd == 'zero':
                self.stop_motion(hard=True)
                self.x_axis.zero_virtual_position()
                self.y_axis.zero_virtual_position()
                self.uart.send('[stepper,zero,ok]')
            elif typ in ('stepper', 'motor') and cmd in ('stop', 'reset'):
                self.stop_and_reset_tracker(zero_virtual=False)
                self.uart.send('[stepper,stop,ok]')
            elif typ in ('stepper', 'motor') and cmd in ('test', 'selftest'):
                self.common_enable.enable()
                self.estop = False
                self.startup_motor_self_test()
                self.uart.send('[stepper,selftest,ok]')
            elif typ in ('stepper', 'motor') and cmd == 'jog' and (len(parts) >= 5):
                axis_name = parts[2].strip().lower()
                try:
                    jog_hz = float(parts[3])
                    jog_ms = int(float(parts[4]))
                    jog_hz = clamp(jog_hz, -450.0, 450.0)
                    jog_ms = int(clamp(jog_ms, 50, 3000))
                    axis = self.x_axis if axis_name == 'x' else self.y_axis
                    self.common_enable.enable()
                    self.estop = False
                    self._jog_axis_blocking(axis, jog_hz, jog_ms)
                    self.uart.send('[stepper,jog,%s,%.0f,%d,ok]' % (axis_name, jog_hz, jog_ms))
                except Exception as e:
                    self.uart.send('[stepper,jog,error,%s]' % str(e))
            elif typ == 'servo' and cmd == 'center':
                self.stop_and_reset_tracker(zero_virtual=False)
                self.uart.send('[stepper,stopped,no_home_reference]')
            elif typ in ('stepper', 'motor', 'servo') and cmd in ('get_state', 'get_angle'):
                self.uart.send('[stepper,state,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%d]' % (self.x_axis.target_hz, self.y_axis.target_hz, self.x_axis.applied_hz, self.y_axis.applied_hz, self.x_axis.virtual_steps, self.y_axis.virtual_steps, self.x_axis.limit_hit, self.y_axis.limit_hit))
            elif typ in ('stepper', 'motor', 'servo') and cmd in ('get_limit', 'get_limits'):
                x_limit_deg = self.x_axis.soft_limit_steps * 360.0 / float(PULSES_PER_REV)
                y_limit_deg = self.y_axis.soft_limit_steps * 360.0 / float(PULSES_PER_REV)
                self.uart.send('[stepper,limit,%.1f,%.1f,%.2f,%.2f]' % (self.x_axis.soft_limit_steps, self.y_axis.soft_limit_steps, x_limit_deg, y_limit_deg))
            elif typ in ('stepper', 'motor', 'servo') and cmd == 'get_pid':
                self.uart.send('[pid,%.6f,0,%.6f,%.6f,0,%.6f,%.1f,%.1f,%.1f,%.1f,%.2f]' % (STATIC_AIM_CFG['x_kp_far'], STATIC_AIM_CFG['x_kd_mid'], STATIC_AIM_CFG['y_kp_far'], STATIC_AIM_CFG['y_kd_mid'], self.x_axis.min_hz, self.y_axis.min_hz, self.x_axis.max_hz, self.y_axis.max_hz, MOTION_CFG['speed_scale']))
            elif typ == 'system' and cmd == 'ping':
                self.uart.send('[system,pong]')
            else:
                self.uart.send('[ack,error,unknown_command,%s,%s]' % (typ, cmd))

    def _apply_base_slider(self, name, raw_value):
        global PLOT_INTERVAL_MS, DIAG_INTERVAL_MS, PLOT_MODE, ENABLE_DIAG_PACKET, ENABLE_PERF_PACKET
        try:
            value = float(raw_value)
        except Exception:
            self.uart.send('[ack,error,bad_value,%s]' % name)
            return
        ok = True
        reset_pid = False
        name = name.strip().lower()
        if name in ('xkp', 'x_kp', 'pankp', 'pan_kp'):
            v = clamp(value, 0.0, 20.0)
            STATIC_AIM_CFG['x_kp_far'] = v
            STATIC_AIM_CFG['x_kp_mid'] = v * 0.79
            STATIC_AIM_CFG['x_kp_fine'] = v * 0.53
            reset_pid = True
        elif name in ('xki', 'x_ki', 'panki', 'pan_ki'):
            self.x_pid.ki = clamp(value, 0.0, 10.0)
            reset_pid = True
        elif name in ('xkd', 'x_kd', 'pankd', 'pan_kd'):
            v = clamp(value, 0.0, 5.0)
            STATIC_AIM_CFG['x_kd_far'] = v * 0.60
            STATIC_AIM_CFG['x_kd_mid'] = v
            STATIC_AIM_CFG['x_kd_fine'] = v * 1.20
            reset_pid = True
        elif name in ('ykp', 'y_kp', 'tiltkp', 'tilt_kp'):
            v = clamp(value, 0.0, 20.0)
            STATIC_AIM_CFG['y_kp_far'] = v
            STATIC_AIM_CFG['y_kp_mid'] = v * 0.81
            STATIC_AIM_CFG['y_kp_fine'] = v * 0.59
            reset_pid = True
        elif name in ('yki', 'y_ki', 'tiltki', 'tilt_ki'):
            self.y_pid.ki = clamp(value, 0.0, 10.0)
            reset_pid = True
        elif name in ('ykd', 'y_kd', 'tiltkd', 'tilt_kd'):
            v = clamp(value, 0.0, 5.0)
            STATIC_AIM_CFG['y_kd_far'] = v * 0.60
            STATIC_AIM_CFG['y_kd_mid'] = v
            STATIC_AIM_CFG['y_kd_fine'] = v * 1.20
            reset_pid = True
        elif name == 'kp':
            v = clamp(value, 0.0, 20.0)
            STATIC_AIM_CFG['x_kp_far'] = v
            STATIC_AIM_CFG['x_kp_mid'] = v * 0.79
            STATIC_AIM_CFG['x_kp_fine'] = v * 0.53
            STATIC_AIM_CFG['y_kp_far'] = v * 0.83
            STATIC_AIM_CFG['y_kp_mid'] = v * 0.67
            STATIC_AIM_CFG['y_kp_fine'] = v * 0.49
            reset_pid = True
        elif name == 'ki':
            self.x_pid.ki = self.y_pid.ki = clamp(value, 0.0, 10.0)
            reset_pid = True
        elif name == 'kd':
            v = clamp(value, 0.0, 5.0)
            STATIC_AIM_CFG['x_kd_far'] = STATIC_AIM_CFG['y_kd_far'] = v * 0.60
            STATIC_AIM_CFG['x_kd_mid'] = STATIC_AIM_CFG['y_kd_mid'] = v
            STATIC_AIM_CFG['x_kd_fine'] = STATIC_AIM_CFG['y_kd_fine'] = v * 1.20
            reset_pid = True
        elif name in ('deadzone', 'dead', 'deadzonepx'):
            v = clamp(value, 0.5, 20.0)
            STATIC_AIM_CFG['x_lock_enter_px'] = STATIC_AIM_CFG['y_lock_enter_px'] = v
            STATIC_AIM_CFG['x_lock_exit_px'] = STATIC_AIM_CFG['y_lock_exit_px'] = max(v + 2.0, v * 1.7)
            reset_pid = True
        elif name in ('xdead', 'x_dead', 'pandead', 'pan_dead'):
            v = clamp(value, 0.5, 20.0)
            STATIC_AIM_CFG['x_lock_enter_px'] = v
            STATIC_AIM_CFG['x_lock_exit_px'] = max(v + 2.0, v * 1.7)
            reset_pid = True
        elif name in ('ydead', 'y_dead', 'tiltdead', 'tilt_dead'):
            v = clamp(value, 0.5, 20.0)
            STATIC_AIM_CFG['y_lock_enter_px'] = v
            STATIC_AIM_CFG['y_lock_exit_px'] = max(v + 2.0, v * 1.7)
            reset_pid = True
        elif name in ('xff', 'x_ff', 'panff'):
            MOTION_CFG['x_ff'] = 0.0
        elif name in ('yff', 'y_ff', 'tiltff'):
            MOTION_CFG['y_ff'] = 0.0
        elif name in ('lead', 'predict', 'leadtime'):
            TRACK_CFG['control_lead_s'] = 0.0
        elif name in ('speedscale', 'speed_scale', 'speed', 'motorscale'):
            MOTION_CFG['speed_scale'] = clamp(value, 0.3, 2.5)
        elif name in ('xnearcap', 'x_near_cap'):
            STATIC_AIM_CFG['x_cap_fine_hz'] = clamp(value, 30.0, self.x_axis.max_hz)
        elif name in ('ynearcap', 'y_near_cap'):
            STATIC_AIM_CFG['y_cap_fine_hz'] = clamp(value, 30.0, self.y_axis.max_hz)
        elif name in ('xmidcap', 'x_mid_cap'):
            STATIC_AIM_CFG['x_cap_mid_hz'] = clamp(value, 30.0, self.x_axis.max_hz)
        elif name in ('ymidcap', 'y_mid_cap'):
            STATIC_AIM_CFG['y_cap_mid_hz'] = clamp(value, 30.0, self.y_axis.max_hz)
        elif name in ('xfarcap', 'x_far_cap'):
            STATIC_AIM_CFG['x_cap_far_hz'] = clamp(value, 30.0, self.x_axis.max_hz)
        elif name in ('yfarcap', 'y_far_cap'):
            STATIC_AIM_CFG['y_cap_far_hz'] = clamp(value, 30.0, self.y_axis.max_hz)
        elif name in ('braketime', 'approach_brake_time'):
            MOTION_CFG['approach_brake_time_s'] = clamp(value, 0.03, 0.6)
        elif name in ('predictstop', 'approach_predict'):
            MOTION_CFG['approach_predict_s'] = clamp(value, 0.0, 0.25)
        elif name in ('approachmaxerr', 'approach_max_error'):
            MOTION_CFG['approach_max_error_px'] = clamp(value, 20.0, 180.0)
        elif name in ('errorfloor', 'error_floor_start'):
            MOTION_CFG['error_floor_start_px'] = clamp(value, 6.0, 80.0)
        elif name in ('xcoastmax', 'x_coast_max'):
            MOTION_CFG['x_coast_max_hz'] = clamp(value, 0.0, self.x_axis.max_hz)
        elif name in ('ycoastmax', 'y_coast_max'):
            MOTION_CFG['y_coast_max_hz'] = clamp(value, 0.0, self.y_axis.max_hz)
        elif name in ('coast1', 'coast_scale_1'):
            MOTION_CFG['coast_scale_1'] = clamp(value, 0.0, 1.2)
        elif name in ('coast2', 'coast_scale_2'):
            MOTION_CFG['coast_scale_2'] = clamp(value, 0.0, 1.2)
        elif name in ('coast3', 'coast_scale_3'):
            MOTION_CFG['coast_scale_3'] = clamp(value, 0.0, 1.2)
        elif name in ('coast4', 'coast_scale_4'):
            MOTION_CFG['coast_scale_4'] = clamp(value, 0.0, 1.2)
        elif name in ('coast5', 'coast_scale_5'):
            MOTION_CFG['coast_scale_5'] = clamp(value, 0.0, 1.2)
        elif name in ('coast6', 'coast_scale_6'):
            MOTION_CFG['coast_scale_6'] = clamp(value, 0.0, 1.2)
        elif name in ('coastpredict', 'coast_predict_stop'):
            MOTION_CFG['coast_predict_stop_s'] = clamp(value, 0.0, 0.25)
        elif name in ('coastouterr', 'coast_outward_error'):
            MOTION_CFG['coast_outward_error_px'] = clamp(value, 5.0, 250.0)
        elif name in ('coastoutvel', 'coast_outward_velocity'):
            MOTION_CFG['coast_outward_velocity_px_s'] = clamp(value, 0.0, 1000.0)
        elif name in ('xcoastoutmax', 'x_coast_out_max'):
            MOTION_CFG['x_coast_out_max_hz'] = clamp(value, 0.0, self.x_axis.max_hz)
        elif name in ('ycoastoutmax', 'y_coast_out_max'):
            MOTION_CFG['y_coast_out_max_hz'] = clamp(value, 0.0, self.y_axis.max_hz)
        elif name in ('coastedgeboost', 'coast_edge_boost'):
            MOTION_CFG['coast_edge_boost'] = clamp(value, 1.0, 1.4)
        elif name in ('coastrefgain', 'coast_reference_gain'):
            MOTION_CFG['coast_reference_gain'] = clamp(value, 0.8, 1.6)
        elif name in ('coastrefmargin', 'coast_reference_margin'):
            MOTION_CFG['coast_reference_margin_hz'] = clamp(value, 0.0, 250.0)
        elif name in ('relaxedstride', 'relaxed_scan_stride'):
            RECT_CFG['full_scan_after_miss'] = int(clamp(round(value), 1, 6))
        elif name in ('recovervar', 'measurement_var_recover_scale'):
            TRACK_CFG['measurement_var_recover_scale'] = clamp(value, 1.0, 4.0)
        elif name in ('lowvar', 'measurement_var_low_scale'):
            TRACK_CFG['measurement_var_low_scale'] = clamp(value, 1.0, 4.0)
        elif name in ('xvelcap', 'max_velocity_x'):
            TRACK_CFG['max_velocity_x_px_s'] = clamp(value, 100.0, 1200.0)
        elif name in ('yvelcap', 'max_velocity_y'):
            TRACK_CFG['max_velocity_y_px_s'] = clamp(value, 100.0, 1200.0)
        elif name in ('xminhz', 'x_min_hz', 'panminhz'):
            self.x_axis.min_hz = clamp(value, 5.0, self.x_axis.max_hz)
        elif name in ('yminhz', 'y_min_hz', 'tiltminhz'):
            self.y_axis.min_hz = clamp(value, 5.0, self.y_axis.max_hz)
        elif name in ('minhz', 'min_hz'):
            v = clamp(value, 5.0, min(self.x_axis.max_hz, self.y_axis.max_hz))
            self.x_axis.min_hz = self.y_axis.min_hz = v
        elif name in ('xmaxhz', 'x_max_hz', 'panmaxhz'):
            self.x_axis.max_hz = clamp(value, self.x_axis.min_hz, 5000.0)
            self.x_pid.output_limit = self.x_axis.max_hz
        elif name in ('ymaxhz', 'y_max_hz', 'tiltmaxhz'):
            self.y_axis.max_hz = clamp(value, self.y_axis.min_hz, 5000.0)
            self.y_pid.output_limit = self.y_axis.max_hz
        elif name in ('maxhz', 'max_hz', 'limit', 'outputlimit', 'output_limit'):
            v = clamp(value, max(self.x_axis.min_hz, self.y_axis.min_hz), 5000.0)
            self.x_axis.max_hz = self.y_axis.max_hz = v
            self.x_pid.output_limit = self.y_pid.output_limit = v
        elif name in ('xaccel', 'x_accel'):
            self.x_axis.accel_hz_s = clamp(value, 20.0, 30000.0)
        elif name in ('yaccel', 'y_accel'):
            self.y_axis.accel_hz_s = clamp(value, 20.0, 30000.0)
        elif name in ('xdecel', 'x_decel'):
            self.x_axis.decel_hz_s = clamp(value, 20.0, 30000.0)
        elif name in ('ydecel', 'y_decel'):
            self.y_axis.decel_hz_s = clamp(value, 20.0, 30000.0)
        elif name in ('accel', 'ramp'):
            v = clamp(value, 20.0, 30000.0)
            self.x_axis.accel_hz_s = self.y_axis.accel_hz_s = v
        elif name == 'decel':
            v = clamp(value, 20.0, 30000.0)
            self.x_axis.decel_hz_s = self.y_axis.decel_hz_s = v
        elif name in ('xreverse', 'x_reverse', 'panreverse'):
            self.x_axis.reverse = bool(value >= 0.5)
            self.x_axis.hard_stop()
            self.x_axis.direction_sign = 0
        elif name in ('yreverse', 'y_reverse', 'tiltreverse'):
            self.y_axis.reverse = bool(value >= 0.5)
            self.y_axis.hard_stop()
            self.y_axis.direction_sign = 0
        elif name in ('xlimitsteps', 'x_limit_steps'):
            self.x_axis.soft_limit_steps = 0.0 if not SOFT_LIMITS_ENABLED else max(0.0, value)
        elif name in ('ylimitsteps', 'y_limit_steps'):
            self.y_axis.soft_limit_steps = 0.0 if not SOFT_LIMITS_ENABLED else max(0.0, value)
        elif name in ('xlimitdeg', 'x_limit_deg', 'panlimitdeg'):
            self.x_axis.soft_limit_steps = 0.0 if not SOFT_LIMITS_ENABLED else max(0.0, value * float(PULSES_PER_REV) / 360.0)
        elif name in ('ylimitdeg', 'y_limit_deg', 'tiltlimitdeg'):
            self.y_axis.soft_limit_steps = 0.0 if not SOFT_LIMITS_ENABLED else max(0.0, value * float(PULSES_PER_REV) / 360.0)
        elif name in ('coast', 'coastframes', 'maxlost'):
            TRACK_CFG['max_coast_frames'] = int(clamp(round(value), 0, 30))
        elif name in ('coasten', 'coast_enable'):
            MOTION_CFG['coast_enable'] = bool(value >= 0.5)
        elif name in ('kalmanq', 'accelnoise'):
            TRACK_CFG['kalman_accel_noise'] = clamp(value, 30.0, 2000.0)
        elif name in ('gate', 'gatetracking', 'trackgate'):
            TRACK_CFG['gate_tracking_px'] = clamp(value, 20.0, 400.0)
        elif name in ('reacquiregate', 'gatereacquire'):
            TRACK_CFG['gate_reacquire_px'] = clamp(value, 40.0, 500.0)
        elif name in ('minarea', 'area'):
            RECT_CFG['min_area_detect'] = clamp(value, 30.0, 12000.0)
        elif name in ('amin', 'aspectmin', 'ratiomin'):
            RECT_CFG['aspect_min'] = clamp(value, 1.0, 2.8)
        elif name in ('amax', 'aspectmax', 'ratiomax'):
            RECT_CFG['aspect_max'] = clamp(value, 1.05, 3.5)
        elif name in ('targetaspect', 'targetratio', 'aspect'):
            RECT_CFG['target_aspect'] = clamp(value, 1.0, 3.0)
        elif name in ('rectsearchth', 'searchth', 'rect_threshold_search'):
            RECT_CFG['threshold_search'] = int(clamp(round(value), 500, 50000))
        elif name in ('recttrackth', 'trackth', 'rect_threshold_track'):
            RECT_CFG['threshold_track'] = int(clamp(round(value), 500, 50000))
        elif name in ('rectcoastth', 'coastth', 'rect_threshold_coast'):
            RECT_CFG['threshold_coast'] = int(clamp(round(value), 500, 50000))
        elif name in ('qhigh', 'qualityhigh'):
            RECT_CFG['quality_high'] = clamp(value, 0.1, 0.95)
        elif name in ('qlow', 'qualitylow'):
            RECT_CFG['quality_low'] = clamp(value, 0.05, RECT_CFG['quality_high'])
        elif name in ('qsearch', 'qualitysearch', 'searchquality'):
            RECT_CFG['quality_search_recover'] = clamp(value, 0.05, 0.95)
        elif name in ('lowring', 'low_stage_min_ring'):
            RECT_CFG['low_stage_min_ring'] = clamp(value, 0.0, 1.0)
        elif name in ('jumpguard', 'jump_guard'):
            TRACK_CFG['jump_guard_px'] = clamp(value, 20.0, 260.0)
        elif name in ('cvangle', 'cv_max_angle_cos'):
            RECT_CFG['cv_max_angle_cos'] = clamp(value, 0.15, 0.7)
        elif name in ('nativeen', 'native_enable'):
            RECT_CFG['native_enable'] = bool(value >= 0.5)
            self.tracker.native_runtime_enabled = RECT_CFG['native_enable']
            self.tracker.native_slow_count = 0
        elif name in ('mincontrast', 'contrast'):
            RECT_CFG['min_contrast'] = clamp(value, -40.0, 120.0)
        elif name in ('roiscalew', 'roi_scale_w'):
            RECT_CFG['roi_scale_w'] = clamp(value, 1.2, 6.0)
        elif name in ('roiscaleh', 'roi_scale_h'):
            RECT_CFG['roi_scale_h'] = clamp(value, 1.2, 6.0)
        elif name in ('roiminw', 'roi_min_w'):
            RECT_CFG['roi_min_w'] = int(clamp(round(value), 40, DETECT_WIDTH))
        elif name in ('roiminh', 'roi_min_h'):
            RECT_CFG['roi_min_h'] = int(clamp(round(value), 40, DETECT_HEIGHT))
        elif name in ('fullscann', 'full_scan_interval'):
            RECT_CFG['full_scan_interval'] = int(clamp(round(value), 1, 120))
        elif name in ('refineevery', 'corner_refine_every'):
            RECT_CFG['corner_refine_every'] = int(clamp(round(value), 1, 20))
        elif name in ('refineenable', 'corner_refine_enable'):
            RECT_CFG['corner_refine_enable'] = bool(value >= 0.5)
        elif name in ('perfen', 'perfenable', 'perf_enable'):
            ENABLE_PERF_PACKET = bool(value >= 0.5)
        elif name in ('plotmode', 'plot_mode'):
            PLOT_MODE = int(clamp(round(value), 0, 2))
            self.uart.send('[plot-clear]')
        elif name in ('plotms', 'plotinterval', 'plot_interval'):
            PLOT_INTERVAL_MS = int(clamp(round(value), 20, 500))
        elif name in ('diagms', 'diaginterval'):
            DIAG_INTERVAL_MS = int(clamp(round(value), 100, 2000))
        elif name in ('diagen', 'diagenable', 'diag_enable'):
            ENABLE_DIAG_PACKET = bool(value >= 0.5)
        else:
            ok = False
        if ok:
            if reset_pid:
                self.reset_pid_memory()
                self._reset_static_aim_memory(clear_command=False)
            self.uart.send('[ack,slider,%s,%s]' % (name, str(value)))
        else:
            self.uart.send('[ack,error,unknown_slider,%s]' % name)

    def update_time(self):
        now = time.ticks_ms()
        dt_ms = max(1, ticks_diff_ms(now, self.last_frame_ms))
        self.last_frame_ms = now
        self.last_dt_ms = dt_ms
        dt = clamp(dt_ms / 1000.0, CONTROL_DT_MIN, CONTROL_DT_MAX)
        inst_fps = 1000.0 / dt_ms
        self.fps = inst_fps if self.fps <= 0.0 else 0.85 * self.fps + 0.15 * inst_fps
        return (now, dt)

    def _base_state_code(self, result):
        if self.estop or not self.tracking_enabled:
            return 4
        if result['state'] == STATE_ACQUIRE:
            return 1
        if result['state'] == STATE_TRACK:
            return 2
        if result['state'] == STATE_COAST:
            return 3
        return 0

    def send_diag(self, result):
        if not ENABLE_DIAG_PACKET or not self.uart.is_ready():
            return
        now = time.ticks_ms()
        if ticks_diff_ms(now, self.last_diag_ms) < DIAG_INTERVAL_MS:
            return
        self.last_diag_ms = now
        raw = result.get('raw_center')
        filt = result.get('filtered_center')
        vx, vy = result.get('velocity', (0.0, 0.0))
        raw_x, raw_y = (-1.0, -1.0) if raw is None else raw
        filt_x, filt_y = (-1.0, -1.0) if filt is None else filt
        self.uart.send('[diag,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.3f,%.1f,%.1f,%.1f,%.1f,%d,%d,%.1f,%.1f,%d,%d,%d,%d,%d,%d,%d]' % (now, self.last_dt_ms, raw_x, raw_y, filt_x, filt_y, vx, vy, float(result.get('confidence', 0.0)), self.x_axis.target_hz, self.y_axis.target_hz, self.x_axis.applied_hz, self.y_axis.applied_hz, self.x_axis.direction_sign, self.y_axis.direction_sign, self.x_axis.virtual_steps, self.y_axis.virtual_steps, self.x_axis.limit_hit, self.y_axis.limit_hit, int(result.get('miss_frames', 0)), int(result.get('relaxed_used', 0)), int(result.get('strict_miss_count', 0)), int(result.get('relaxed_recover_count', 0)), int(result.get('hard_miss_count', 0))))

    def send_perf(self, result):
        if not ENABLE_PERF_PACKET or not self.uart.is_ready():
            return
        now = time.ticks_ms()
        if ticks_diff_ms(now, self.last_perf_ms) < PERF_INTERVAL_MS:
            return
        self.last_perf_ms = now
        self.uart.send('[perf,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d]' % (self.perf_capture_ms, float(result.get('detect_ms', 0.0)), float(result.get('associate_ms', 0.0)), float(result.get('refine_ms', 0.0)), self.perf_control_ms, self.perf_display_ms, self.perf_total_ms, int(result.get('roi_area', 0)), int(result.get('full_scan', 0)), int(result.get('candidate_count', 0)), self.state_code(result)))

    def reset_laser_control(self, clear_fault=False):
        self.control_arm_count = 0
        self.previous_error_norm = None
        self.divergence_count = 0
        self.reset_pid_memory()
        self.filtered_err_x = None
        self.filtered_err_y = None
        self.filtered_rel_vx = 0.0
        self.filtered_rel_vy = 0.0
        self.invalid_control_frames = 0
        self.x_precision_hold = False
        self.y_precision_hold = False
        self.x_stop_confirm_count = 0
        self.y_stop_confirm_count = 0
        self.last_x_zone = 0
        self.last_y_zone = 0
        self.stable_align_count = 0
        self._reset_static_aim_memory(clear_command=False)
        self._reset_response_guard()
        if clear_fault:
            self.direction_fault = False
            self.direction_fault_reported = False

    def state_code(self, result):
        return self._base_state_code(result)

    def _measure_error(self, result):
        rect = result.get('center')
        laser = self.current_laser_pos
        if rect is None or laser is None or (not self.laser.locked):
            return None
        return (float(rect[0] - laser[0]), float(rect[1] - laser[1]))

    def _reset_static_aim_memory(self, clear_command=False):
        self.filtered_err_x = None
        self.filtered_err_y = None
        self.x_error_derivative = 0.0
        self.y_error_derivative = 0.0
        self.last_err_x = None
        self.last_err_y = None
        self.x_aim_locked = False
        self.y_aim_locked = False
        self.x_aim_lock_count = 0
        self.y_aim_lock_count = 0
        self.static_miss_frames = 0
        self.aim_mode = 'WAIT'
        if clear_command:
            self.last_x_target_hz = 0.0
            self.last_y_target_hz = 0.0
            self.last_track_x_hz = 0.0
            self.last_track_y_hz = 0.0

    def _reset_response_guard(self):
        self.x_response_start_ms = None
        self.y_response_start_ms = None
        self.x_response_start_error = 0.0
        self.y_response_start_error = 0.0
        self.x_response_command_sign = 0
        self.y_response_command_sign = 0
        self.x_response_error_sign = 0
        self.y_response_error_sign = 0
        self.x_recovery_start_ms = None
        self.y_recovery_start_ms = None
        self.x_response_fail_count = 0
        self.y_response_fail_count = 0

    def _guard_axis_response(self, axis_name, raw_error, target_hz):
        """高频无视觉响应时临时降速；不锁死，恢复期结束后自动重试。"""
        now = time.ticks_ms()
        prefix = 'x_' if axis_name == 'x' else 'y_'
        recovery_attr = prefix + 'recovery_start_ms'
        start_attr = prefix + 'response_start_ms'
        start_error_attr = prefix + 'response_start_error'
        command_sign_attr = prefix + 'response_command_sign'
        error_sign_attr = prefix + 'response_error_sign'
        fail_count_attr = prefix + 'response_fail_count'
        recovery_cap = float(RESPONSE_GUARD_CFG[prefix + 'recovery_cap_hz'])

        recovery_start = getattr(self, recovery_attr)
        if recovery_start is not None:
            if ticks_diff_ms(now, recovery_start) < int(RESPONSE_GUARD_CFG['recovery_ms']):
                return clamp(float(target_hz), -recovery_cap, recovery_cap), True
            setattr(self, recovery_attr, None)
            setattr(self, start_attr, None)

        abs_error = abs(float(raw_error))
        command = float(target_hz)
        command_sign = sign_of(command)
        error_sign = sign_of(raw_error)
        if (
            abs_error < float(RESPONSE_GUARD_CFG['min_error_px'])
            or abs(command) < float(RESPONSE_GUARD_CFG['min_command_hz'])
        ):
            setattr(self, start_attr, None)
            return command, False

        start_ms = getattr(self, start_attr)
        if (
            start_ms is None
            or command_sign != int(getattr(self, command_sign_attr))
            or error_sign != int(getattr(self, error_sign_attr))
        ):
            setattr(self, start_attr, now)
            setattr(self, start_error_attr, abs_error)
            setattr(self, command_sign_attr, command_sign)
            setattr(self, error_sign_attr, error_sign)
            return command, False

        elapsed = ticks_diff_ms(now, start_ms)
        start_error = float(getattr(self, start_error_attr))
        required = max(
            float(RESPONSE_GUARD_CFG['min_improve_px']),
            start_error * float(RESPONSE_GUARD_CFG['min_improve_ratio']),
        )
        insufficient = start_error - abs_error < required
        growing_fast = (
            elapsed >= int(RESPONSE_GUARD_CFG['fast_fail_ms'])
            and abs_error >= start_error + float(RESPONSE_GUARD_CFG['growth_fail_px'])
        )
        timed_out = elapsed >= int(RESPONSE_GUARD_CFG['observe_ms']) and insufficient
        if growing_fast or timed_out:
            count = int(getattr(self, fail_count_attr)) + 1
            setattr(self, fail_count_attr, count)
            setattr(self, recovery_attr, now)
            setattr(self, start_attr, None)
            print('MOTION RESPONSE %s #%d: err %.1f->%.1f, fallback %.0fHz' % (
                axis_name.upper(), count, start_error, abs_error, recovery_cap,
            ))
            return clamp(command, -recovery_cap, recovery_cap), True

        if elapsed >= int(RESPONSE_GUARD_CFG['observe_ms']):
            setattr(self, start_attr, now)
            setattr(self, start_error_attr, abs_error)
        return command, False

    def _apply_visual_edge_guard(self, x_target, y_target):
        laser = self.current_laser_pos
        if laser is None:
            return (0.0, 0.0)
        lx, ly = laser
        if lx <= LASER_EDGE_GUARD_X_PX and x_target < 0.0:
            x_target = 0.0
        elif lx >= CAMERA_WIDTH - LASER_EDGE_GUARD_X_PX and x_target > 0.0:
            x_target = 0.0
        if ly <= LASER_EDGE_GUARD_Y_PX and y_target < 0.0:
            y_target = 0.0
        elif ly >= CAMERA_HEIGHT - LASER_EDGE_GUARD_Y_PX and y_target > 0.0:
            y_target = 0.0
        return (x_target, y_target)

    @staticmethod
    def _static_zone(axis_name, abs_error):
        prefix = 'x_' if axis_name == 'x' else 'y_'
        if abs_error > float(STATIC_AIM_CFG['mid_error_px']):
            zone = 3
            suffix = 'far'
            alpha = float(STATIC_AIM_CFG['alpha_far'])
        elif abs_error > float(STATIC_AIM_CFG['fine_error_px']):
            zone = 2
            suffix = 'mid'
            alpha = float(STATIC_AIM_CFG['alpha_mid'])
        else:
            zone = 1
            suffix = 'fine'
            alpha = float(STATIC_AIM_CFG['alpha_fine'])
        return (
            zone,
            float(STATIC_AIM_CFG[prefix + 'kp_' + suffix]),
            float(STATIC_AIM_CFG[prefix + 'kd_' + suffix]),
            float(STATIC_AIM_CFG[prefix + 'cap_' + suffix + '_hz']),
            alpha,
        )

    def _static_axis_command(self, axis_name, raw_error, dt):
        """静止靶单轴三段PD；显著误差首帧直接出步，小误差用迟滞稳定停步。"""
        raw_error = float(raw_error)
        dt = clamp(safe_float(dt, 0.033), CONTROL_DT_MIN, CONTROL_DT_MAX)
        if axis_name == 'x':
            filter_attr = 'filtered_err_x'
            derivative_attr = 'x_error_derivative'
            previous_attr = 'last_err_x'
            locked_attr = 'x_aim_locked'
            count_attr = 'x_aim_lock_count'
            enter = float(STATIC_AIM_CFG['x_lock_enter_px'])
            exit_error = float(STATIC_AIM_CFG['x_lock_exit_px'])
        else:
            filter_attr = 'filtered_err_y'
            derivative_attr = 'y_error_derivative'
            previous_attr = 'last_err_y'
            locked_attr = 'y_aim_locked'
            count_attr = 'y_aim_lock_count'
            enter = float(STATIC_AIM_CFG['y_lock_enter_px'])
            exit_error = float(STATIC_AIM_CFG['y_lock_exit_px'])

        filtered = getattr(self, filter_attr)
        previous = getattr(self, previous_attr)
        zone, kp, kd, cap_hz, alpha = self._static_zone(axis_name, abs(raw_error))
        if filtered is None or previous is None:
            filtered = raw_error
            previous = raw_error
            derivative = 0.0
        else:
            # 误差突然变大或跨过中心时直接提高响应，避免滤波造成“识别到了却不动”。
            if raw_error * filtered <= 0.0 or abs(raw_error) > abs(filtered) + 5.0:
                alpha = max(alpha, 0.82)
            filtered += alpha * (raw_error - filtered)
            raw_derivative = (filtered - previous) / max(0.004, dt)
            da = float(STATIC_AIM_CFG['derivative_alpha'])
            derivative = getattr(self, derivative_attr) + da * (
                raw_derivative - getattr(self, derivative_attr)
            )

        setattr(self, filter_attr, filtered)
        setattr(self, derivative_attr, derivative)
        setattr(self, previous_attr, filtered)

        locked = bool(getattr(self, locked_attr))
        if locked:
            if abs(raw_error) <= exit_error:
                return 0.0, 0
            setattr(self, locked_attr, False)
            setattr(self, count_attr, 0)

        if abs(raw_error) <= enter:
            count = int(getattr(self, count_attr)) + 1
            setattr(self, count_attr, count)
            if count >= int(STATIC_AIM_CFG['lock_confirm_frames']):
                setattr(self, locked_attr, True)
            return 0.0, 0
        setattr(self, count_attr, 0)

        target = (kp * filtered + kd * derivative) * float(MOTION_CFG['speed_scale'])
        # D项只能帮助减速，不能在尚未越过中心时反向驱动。
        if target * raw_error < 0.0:
            target = 0.0
        return clamp(target, -cap_hz, cap_hz), zone

    def _hold_last_static_command(self, dt):
        self.static_miss_frames += 1
        miss = self.static_miss_frames
        if miss == 1:
            scale = float(STATIC_AIM_CFG['miss_scale_1'])
        elif miss == 2:
            scale = float(STATIC_AIM_CFG['miss_scale_2'])
        else:
            self.aim_mode = 'WAIT'
            self.decelerate_motion(dt)
            self.last_x_target_hz = 0.0
            self.last_y_target_hz = 0.0
            self.last_track_x_hz = 0.0
            self.last_track_y_hz = 0.0
            return

        self.aim_mode = 'HOLD%d' % miss
        self.last_x_target_hz = self.x_axis.set_target_hz(self.last_track_x_hz * scale)
        self.last_y_target_hz = self.y_axis.set_target_hz(self.last_track_y_hz * scale)
        self.x_axis.update(dt)
        self.y_axis.update(dt)

    def update_tracking_control(self, result, dt):
        """固定跑道静止靶：只用当前真实测量闭环，不使用运动目标预测和速度前馈。"""
        global LASER_X_REVERSE, LASER_Y_REVERSE
        dt = clamp(safe_float(dt, 0.033), CONTROL_DT_MIN, CONTROL_DT_MAX)
        if self.estop:
            self.stop_motion(hard=True)
            self.aim_mode = 'ESTOP'
            return

        measured = self._measure_error(result)
        real_rect = result.get('candidate') is not None and str(result.get('source', '')).startswith('DETECT')
        real_laser = self.laser.locked and self.laser.last_source in ('CORE', 'HALO')
        if result.get('state') != STATE_TRACK or measured is None or not real_rect or not real_laser:
            self._hold_last_static_command(dt)
            return

        if self.static_miss_frames > int(STATIC_AIM_CFG['miss_hold_frames']):
            # 长时间无真实测量后丢弃旧滤波值，恢复首帧直接响应。
            self._reset_static_aim_memory(clear_command=False)
        self.static_miss_frames = 0
        self.invalid_control_frames = 0
        self.control_arm_count = 1
        raw_err_x, raw_err_y = measured
        x_target, self.last_x_zone = self._static_axis_command('x', raw_err_x, dt)
        y_target, self.last_y_zone = self._static_axis_command('y', raw_err_y, dt)

        if self.x_aim_locked and self.y_aim_locked:
            self.aim_mode = 'LOCKED'
        else:
            zone = max(int(self.last_x_zone), int(self.last_y_zone))
            self.aim_mode = 'FAST' if zone >= 3 else ('APPROACH' if zone == 2 else 'FINE')

        x_target, y_target = self._apply_visual_edge_guard(x_target, y_target)
        if LASER_X_REVERSE:
            x_target = -x_target
        if LASER_Y_REVERSE:
            y_target = -y_target

        x_target, x_recovery = self._guard_axis_response('x', raw_err_x, x_target)
        y_target, y_recovery = self._guard_axis_response('y', raw_err_y, y_target)
        if x_recovery or y_recovery:
            suffix = ('X' if x_recovery else '') + ('Y' if y_recovery else '')
            self.aim_mode = 'TORQUE_' + suffix

        # 只保留StepperAxis的一层物理加减速，避免旧版双重斜率限制造成延迟。
        self.last_x_target_hz = self.x_axis.set_target_hz(x_target)
        self.last_y_target_hz = self.y_axis.set_target_hz(y_target)
        self.last_track_x_hz = self.last_x_target_hz
        self.last_track_y_hz = self.last_y_target_hz
        self.x_axis.update(dt)
        self.y_axis.update(dt)

    def update_coast_control(self, result, dt):
        self.control_arm_count = 0
        self._hold_last_static_command(dt)

    def apply_slider(self, name, raw_value):
        global LASER_X_REVERSE, LASER_Y_REVERSE, LASER_CONTROL_ARM_FRAMES, PLOT_MODE
        n = name.strip().lower()
        try:
            value = float(raw_value)
        except Exception:
            self.uart.send('[ack,error,bad_value,%s]' % name)
            return
        if n in ('laserxreverse', 'laser_x_reverse'):
            LASER_X_REVERSE = bool(value >= 0.5)
            self.stop_motion(hard=True)
            self.reset_laser_control(clear_fault=True)
            self.uart.send('[ack,slider,%s,%d]' % (name, 1 if LASER_X_REVERSE else 0))
            return
        if n in ('laseryreverse', 'laser_y_reverse'):
            LASER_Y_REVERSE = bool(value >= 0.5)
            self.stop_motion(hard=True)
            self.reset_laser_control(clear_fault=True)
            self.uart.send('[ack,slider,%s,%d]' % (name, 1 if LASER_Y_REVERSE else 0))
            return
        if n in ('clearfault', 'directionfaultreset'):
            self.reset_laser_control(clear_fault=True)
            self.uart.send('[ack,slider,%s,0]' % name)
            return
        if n in ('laseractive', 'laser_active', 'laseractivelevel'):
            level = self.laser_output.set_active_level(value)
            self.laser.reset()
            self.current_laser_pos = None
            self.control_arm_count = 0
            self.stop_motion(hard=True)
            self.uart.send('[ack,slider,%s,%d]' % (name, level))
            return
        if n in ('laseron', 'laser_on'):
            if value >= 0.5:
                self.laser_output.on()
            else:
                self.laser_output.off()
                self.laser.reset()
                self.current_laser_pos = None
                self.stop_motion(hard=True)
            self.uart.send('[ack,slider,%s,%d]' % (name, 1 if self.laser_output.enabled else 0))
            return
        if n in ('laserarmframes', 'laser_arm_frames'):
            LASER_CONTROL_ARM_FRAMES = int(clamp(round(value), 1, 30))
            self.control_arm_count = 0
            self.uart.send('[ack,slider,%s,%d]' % (name, LASER_CONTROL_ARM_FRAMES))
            return
        if n in ('plotmode', 'plot_mode'):
            PLOT_MODE = int(clamp(round(value), 0, 3))
            self.uart.send('[plot-clear]')
            self.uart.send('[ack,slider,%s,%d]' % (name, PLOT_MODE))
            return
        mapping = {'laserlmin': ('core_l_min', 0, 100), 'laserlmax': ('core_l_max', 0, 100), 'laseramin': ('core_a_min', -128, 127), 'laseramax': ('core_a_max', -128, 127), 'laserbmin': ('core_b_min', -128, 127), 'laserbmax': ('core_b_max', -128, 127), 'laserhalolmin': ('halo_l_min', 0, 100), 'laserhaloamin': ('halo_a_min', -128, 127), 'laserhalobmin': ('halo_b_min', -128, 127), 'laserminarea': ('area_min', 1, 200), 'lasermaxarea': ('area_max', 2, 1000), 'lasermindensity': ('min_density', 0.0, 1.0), 'lasermaxw': ('max_w_det', 2, 100), 'lasermaxh': ('max_h_det', 2, 100), 'lasermaxaspect': ('max_aspect', 1.0, 10.0), 'lasergate': ('gate_locked_px_det', 5, 100), 'laseracquirejump': ('acquire_jump_px_det', 2, 80), 'lasermaxcandidates': ('max_acquire_candidates', 1, 50), 'laserambiguity': ('ambiguity_margin', 0.0, 0.5), 'laserrectmargin': ('rect_roi_margin_det', 5, 160), 'laserposalpha': ('position_alpha', 0.05, 1.0), 'laservelalpha': ('velocity_alpha', 0.05, 1.0)}
        aliases = {'laser_l_min': 'laserlmin', 'laser_l_max': 'laserlmax', 'laser_a_min': 'laseramin', 'laser_a_max': 'laseramax', 'laser_b_min': 'laserbmin', 'laser_b_max': 'laserbmax', 'laser_area_min': 'laserminarea', 'laser_area_max': 'lasermaxarea', 'laser_min_density': 'lasermindensity', 'laser_gate': 'lasergate', 'laser_acquire_jump': 'laseracquirejump', 'laser_max_candidates': 'lasermaxcandidates', 'laser_ambiguity': 'laserambiguity', 'laser_rect_margin': 'laserrectmargin'}
        n2 = aliases.get(n, n)
        if n2 in mapping:
            key, low, high = mapping[n2]
            v = clamp(value, low, high)
            if key in ('area_min', 'area_max', 'max_acquire_candidates'):
                v = int(round(v))
            LASER_CFG[key] = v
            self.laser.reset()
            self.current_laser_pos = None
            self.control_arm_count = 0
            self.stop_motion(hard=True)
            self.uart.send('[ack,slider,%s,%s]' % (name, str(v)))
            return
        self._apply_base_slider(name, raw_value)

    def send_plot(self, result):
        now = time.ticks_ms()
        if ticks_diff_ms(now, self.last_plot_ms) < PLOT_INTERVAL_MS:
            return
        self.last_plot_ms = now
        state_code = self.state_code(result)
        rect = result.get('center')
        laser = self.current_laser_pos
        if rect is None or laser is None:
            raw_err_x = raw_err_y = 0.0
            rect_x = rect_y = 0.0
        else:
            rect_x, rect_y = rect
            raw_err_x = float(rect_x - laser[0])
            raw_err_y = float(rect_y - laser[1])
        if PLOT_MODE == 1:
            source_code = {'NONE': 0, 'CORE': 1, 'HALO': 2, 'PREDICT': 3}.get(str(self.laser.last_source), 0)
            self.uart.send('[plot,%d,%d,%d,%d,%d,%.3f,%d,%d,%d,%.2f]' % (int(self.last_dt_ms), int(result.get('miss_frames', 0)), int(self.laser.lost_frames), int(self.laser.raw_blob_count), int(self.laser.candidate_count), float(self.laser.confidence), int(source_code), int(self.control_arm_count), int(state_code), self.fps))
            return
        if PLOT_MODE == 2:
            self.uart.send('[plot,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%d,%d,%.2f]' % (self.x_axis.target_hz, self.y_axis.target_hz, self.x_axis.applied_hz, self.y_axis.applied_hz, self.x_axis.virtual_steps, self.y_axis.virtual_steps, int(self.x_axis.limit_hit), int(self.y_axis.limit_hit), int(state_code), self.fps))
            return
        if PLOT_MODE == 3:
            filt_x = raw_err_x if self.filtered_err_x is None else self.filtered_err_x
            filt_y = raw_err_y if self.filtered_err_y is None else self.filtered_err_y
            source_code = {'NONE': 0, 'CORE': 1, 'HALO': 2, 'PREDICT': 3}.get(str(self.laser.last_source), 0)
            self.uart.send('[plot,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.3f,%d,%d,%.2f]' % (raw_err_x, raw_err_y, float(filt_x), float(filt_y), float(self.filtered_rel_vx), float(self.filtered_rel_vy), float(self.laser.confidence), int(source_code), int(state_code), self.fps))
            return
        self.uart.send('[plot,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%.2f]' % (raw_err_x, raw_err_y, self.x_axis.target_hz, self.y_axis.target_hz, self.x_axis.applied_hz, self.y_axis.applied_hz, rect_x, rect_y, state_code, self.fps))

    def send_laser_packet(self):
        if not self.uart.is_ready():
            return
        now = time.ticks_ms()
        if ticks_diff_ms(now, self.last_laser_packet_ms) < LASER_PACKET_INTERVAL_MS:
            return
        self.last_laser_packet_ms = now
        if self.current_laser_pos is None:
            lx = ly = -1.0
        else:
            lx, ly = self.current_laser_pos
        vx, vy = self.laser.velocity
        self.uart.send('[laser,%.2f,%.2f,%.2f,%.2f,%.3f,%d,%.1f,%.3f,%d,%d]' % (lx, ly, vx, vy, self.laser.confidence, self.laser.candidate_count, self.laser.last_area, self.laser.last_density, self.laser.lost_frames, 1 if self.laser_output.enabled else 0))

    def draw(self, img, result):
        for c in (result.get('candidates') or [])[:12]:
            try:
                x, y, w, h = c['box_det']
                img.draw_rectangle(int(x), int(y), int(w), int(h), color=(40, 120, 255), thickness=1)
            except Exception:
                pass
        state_color = {STATE_TRACK: (0, 255, 0), STATE_ACQUIRE: (255, 255, 0), STATE_COAST: (255, 140, 0), STATE_LOST: (255, 0, 0), STATE_SEARCH: (255, 0, 0)}.get(result.get('state'), (255, 255, 0))
        corners = result.get('corners_det')
        if corners is not None:
            pts = [(int(round(p[0])), int(round(p[1]))) for p in order_corners(corners)]
            for i in range(4):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % 4]
                img.draw_line(x0, y0, x1, y1, color=state_color, thickness=2)
            for px, py in pts:
                img.draw_circle(px, py, 2, color=(0, 255, 255), thickness=1)
        elif result.get('box_det') is not None:
            x, y, w, h = result['box_det']
            img.draw_rectangle(int(x), int(y), int(w), int(h), color=state_color, thickness=2)
        cx_det = int(round(logical_to_detect_x(self.cx0)))
        cy_det = int(round(logical_to_detect_y(self.cy0)))
        img.draw_cross(cx_det, cy_det, color=(0, 80, 255), size=6, thickness=1)
        rect = result.get('center')
        laser = self.current_laser_pos
        rect_det = None
        laser_det = None
        if rect is not None:
            rect_det = (int(round(logical_to_detect_x(rect[0]))), int(round(logical_to_detect_y(rect[1]))))
            img.draw_cross(rect_det[0], rect_det[1], color=(0, 255, 0), size=7, thickness=2)
        for cand in self.laser.last_candidates_det[:8]:
            cx_c = int(round(cand[0]))
            cy_c = int(round(cand[1]))
            img.draw_circle(cx_c, cy_c, 4, color=(255, 120, 0), thickness=1)
        if laser is not None:
            laser_det = (int(round(logical_to_detect_x(laser[0]))), int(round(logical_to_detect_y(laser[1]))))
            img.draw_cross(laser_det[0], laser_det[1], color=(255, 0, 0), size=7, thickness=2)
        if rect_det is not None and laser_det is not None:
            img.draw_line(laser_det[0], laser_det[1], rect_det[0], rect_det[1], color=(255, 0, 255), thickness=1)
        err = self._measure_error(result)
        if err is None:
            ex = ey = 0.0
        else:
            ex, ey = err
        line1 = '%s AIM=%s R=%d L=%d/%d' % (STATE_NAME[result.get('state', STATE_SEARCH)], self.aim_mode, 1 if rect is not None else 0, 1 if laser is not None else 0, 1 if self.laser.locked else 0)
        line2 = 'e=(%.0f,%.0f) Hz=(%.0f,%.0f) Z=%d/%d RC=%d LC=%d/%d A=%d IO=%d/%d fps=%.1f' % (ex, ey, self.x_axis.applied_hz, self.y_axis.applied_hz, int(self.last_x_zone), int(self.last_y_zone), int(result.get('candidate_count', 0)), int(self.laser.candidate_count), int(self.laser.raw_blob_count), int(self.laser.ambiguous), 1 if self.laser_output.enabled else 0, int(self.laser_output.active_level), self.fps)
        try:
            img.draw_string_advanced(2, 2, 13, line1, color=(255, 255, 255))
            img.draw_string_advanced(2, 18, 13, line2, color=(255, 255, 255))
        except Exception:
            pass

    def print_status(self, result):
        now = time.ticks_ms()
        if ticks_diff_ms(now, self.last_status_ms) < STATUS_INTERVAL_MS:
            return
        self.last_status_ms = now
        rect = result.get('center')
        laser = self.current_laser_pos
        if rect is None or laser is None:
            print('STAT state=%s aim=%s rect=%d laser=%d rectCand=%d laserCand=%d/%d core=%d halo=%d amb=%d outHz=(%.0f,%.0f) resp=%d/%d lim=%d/%d det=%s io=%d/A%d ferr=%d fps=%.1f' % (STATE_NAME[result.get('state', STATE_SEARCH)], self.aim_mode, 1 if rect is not None else 0, 1 if laser is not None else 0, int(result.get('candidate_count', 0)), int(self.laser.candidate_count), int(self.laser.raw_blob_count), int(self.laser.core_blob_count), int(self.laser.halo_blob_count), int(self.laser.ambiguous), self.x_axis.applied_hz, self.y_axis.applied_hz, int(self.x_response_fail_count), int(self.y_response_fail_count), int(self.x_axis.limit_hit), int(self.y_axis.limit_hit), str(self.tracker.last_detector_mode), 1 if self.laser_output.enabled else 0, int(self.laser_output.active_level), int(self.laser.find_error_count), self.fps))
        else:
            print('STAT state=%s aim=%s err=(%.1f,%.1f) rect=(%.1f,%.1f) laser=(%.1f,%.1f) rectCand=%d laserCand=%d/%d src=%s amb=%d outHz=(%.0f,%.0f) resp=%d/%d lim=%d/%d det=%s io=%d/A%d fps=%.1f' % (STATE_NAME[result.get('state', STATE_SEARCH)], self.aim_mode, rect[0] - laser[0], rect[1] - laser[1], rect[0], rect[1], laser[0], laser[1], int(result.get('candidate_count', 0)), int(self.laser.candidate_count), int(self.laser.raw_blob_count), str(self.laser.last_source), int(self.laser.ambiguous), self.x_axis.applied_hz, self.y_axis.applied_hz, int(self.x_response_fail_count), int(self.y_response_fail_count), int(self.x_axis.limit_hit), int(self.y_axis.limit_hit), str(self.tracker.last_detector_mode), 1 if self.laser_output.enabled else 0, int(self.laser_output.active_level), self.fps))

    def startup_motor_self_test(self):
        """仅由串口 [motor,test] 触发，不在程序启动时自动运动。"""
        print('--- D36A manual motor self-test begin ---')
        sequence = (('X+', self.x_axis, +SELF_TEST_HZ), ('X-', self.x_axis, -SELF_TEST_HZ), ('Y+', self.y_axis, +SELF_TEST_HZ), ('Y-', self.y_axis, -SELF_TEST_HZ))
        for label, axis, hz in sequence:
            print('SELFTEST %s %.0fHz %dms' % (label, hz, SELF_TEST_RUN_MS))
            self._jog_axis_blocking(axis, hz, SELF_TEST_RUN_MS)
        self.stop_motion(hard=True)
        self.x_axis.zero_virtual_position()
        self.y_axis.zero_virtual_position()
        self.reset_laser_control(clear_fault=True)
        print('--- D36A manual motor self-test end ---')

    def recover_bad_frame(self, stage, error):
        """坏帧/临时相机异常只复位本帧状态，不再让外层finally结束整个程序。"""
        self.frame_error_count += 1
        now = time.ticks_ms()
        if self.frame_error_count <= 3 or ticks_diff_ms(now, self.last_frame_error_ms) >= 1000:
            self.last_frame_error_ms = now
            print('FRAME RECOVERY [%s] #%d: %s' % (stage, self.frame_error_count, str(error)))
            try:
                sys.print_exception(error)
            except Exception:
                pass
        self.stop_motion(hard=True)
        self.current_laser_pos = None
        self.laser.reset()
        if stage == 'rect':
            self.tracker.reset()
        self.reset_laser_control(clear_fault=True)
        try:
            gc.collect()
        except Exception:
            pass

    def run(self):
        sensor = None
        display_ok = False
        display_initialized = False
        try:
            print('=' * 78)
            print('K230 D36A STATIC-LANE TARGETING V8.1')
            print('Closed loop: rectangle center - laser center')
            print('X STEP/DIR GPIO42/26; Y STEP/DIR GPIO43/34')
            print('UART3 GPIO32/33; laser TTL GPIO35')
            print('D36A EN1/EN2 must remain tied to D36A board 5V')
            print('Static target control: persistent identity + torque fallback + auto retry')
            print('=' * 78)
            print('Startup motor self-test disabled; use [motor,test] manually')
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
                display_initialized = True
                print('Display ready: ST7701 640x480; IDE mirror=%d' % (1 if DISPLAY_TO_IDE else 0))
            MediaManager.init()
            sensor.run()
            time.sleep_ms(300)
            print("Camera channels ready")
            for _ in range(30):
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
                print('Camera ready; RECT ch%d RGB888 + LASER ch%d RGB565; laser GPIO35 ON; searching rectangle and laser' % (RECT_CHANNEL, LASER_CHANNEL))
            if self.uart.is_ready():
                self.uart.send('[system,ready,k230_d36a_static_lane_targeting_v8_1]')
                self.uart.send('[display,8,8,CH1/2 rect-laser error; CH3/4 target Hz; CH5/6 output Hz,14]')
                self.uart.send('[display,8,26,plotMode0 control; 1 loss/source; 2 stepper; 3 filter,14]')
            frame_count = 0
            display_x = int((640 - DETECT_WIDTH) / 2)
            display_y = int((480 - DETECT_HEIGHT) / 2)
            while True:
                os.exitpoint()
                loop_start_us = time.ticks_us()
                now, dt = self.update_time()
                capture_start_us = time.ticks_us()
                if not getattr(sensor, 'run', None):
                    continue
                try:
                    img = sensor.snapshot(chn=RECT_CHANNEL)
                    img_np = img.to_numpy_ref()
                    result = self.tracker.step(img, img_np, dt)
                except Exception as frame_error:
                    self.recover_bad_frame('rect', frame_error)
                    time.sleep_ms(2)
                    continue
                self.perf_capture_ms = perf_ms(capture_start_us)
                if self.laser_output.enabled:
                    try:
                        laser_img = sensor.snapshot(chn=LASER_CHANNEL)
                        laser_np = laser_img.to_numpy_ref()
                        self.current_laser_pos = self.laser.detect(
                            laser_img, laser_np, result.get('box_det'),
                            result.get('center'), dt,
                        )
                    except Exception as laser_error:
                        print('laser detector recovered from bad frame:', laser_error)
                        try:
                            sys.print_exception(laser_error)
                        except Exception:
                            pass
                        self.laser.reset()
                        self.current_laser_pos = None
                else:
                    self.laser.reset()
                    self.current_laser_pos = None
                control_start_us = time.ticks_us()
                try:
                    if self.tracking_enabled and (not self.estop):
                        if result.get('state') == STATE_TRACK:
                            self.update_tracking_control(result, dt)
                        else:
                            # COAST/LOST/SEARCH统一短桥接后平滑停步，避免每帧清空状态。
                            self.update_coast_control(result, dt)
                    else:
                        self.stop_motion(hard=True)
                        self._reset_static_aim_memory(clear_command=True)
                except Exception as control_error:
                    self.recover_bad_frame('control', control_error)
                self.perf_control_ms = perf_ms(control_start_us)
                try:
                    self.handle_uart()
                    self.send_plot(result)
                    self.send_laser_packet()
                    self.send_diag(result)
                    self.print_status(result)
                except Exception as io_error:
                    # 串口/诊断失败不能终止视觉与电机闭环。
                    if frame_count % 30 == 0:
                        print('telemetry skipped:', io_error)
                self.perf_display_ms = 0.0
                frame_count += 1
                if display_ok and frame_count % max(1, DISPLAY_EVERY_N_FRAMES) == 0:
                    display_start_us = time.ticks_us()
                    try:
                        self.draw(img, result)
                        # 直接居中显示低延迟图像，避免额外缩放和大帧分配。
                        Display.show_image(img, x=display_x, y=display_y)
                        self.perf_display_ms = perf_ms(display_start_us)
                    except Exception as display_error:
                        # 显示故障只关闭显示分支，追踪继续运行。
                        display_ok = False
                        print('display disabled after runtime error:', display_error)
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
            print('user stop')
        except BaseException as e:
            print('Exception:', e)
            try:
                sys.print_exception(e)
            except Exception:
                pass
        finally:
            self.stop_motion(hard=True)
            self.laser_output.off()
            self.emergency_stop()
            if isinstance(sensor, Sensor):
                try:
                    sensor.stop()
                except Exception:
                    pass
            if display_initialized:
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
            self.laser_output.deinit()
            print('program exited; STEP PWM 0; laser GPIO35 OFF')
if __name__ == "__main__":
    K230D36AStaticLaneTargetingV81().run()
