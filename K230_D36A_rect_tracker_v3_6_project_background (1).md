# K230 + D36A 步进云台矩形追踪项目背景与 v3.6 最终方案

> 作用：在新对话中作为完整项目背景使用。  
> 当前最终代码：`main_k230_d36a_rect_track_uart_v3_6_final_rect_tracker.py`。  
> 备用文件名：`main_final_rect_tracker.py`。  
> 目标：K230 识别黑色矩形方框中心，并通过 D36A 两轴步进驱动板控制二维云台，让目标中心保持在画面中心。

---

## 1. 当前硬件结构

### 1.1 主控与视觉

- 主控：K230 / CanMV MicroPython。
- 实测固件信息：`CanMV v1.4.3(based on Micropython e00a144)`。
- 摄像头：GC2093 CSI2，日志显示输入能力为 `1280x960@90`。
- 当前处理图像：`640×480 RGB888`。
- 显示：优先虚拟显示到 IDE，失败时不中断追踪。

### 1.2 D36A 步进云台接线

| 功能 | K230 引脚 | D36A |
|---|---:|---|
| X轴 STEP | GPIO42 | STEP1 / X STEP |
| X轴 DIR | GPIO26 | DIR1 / X DIR |
| Y轴 STEP | GPIO43 | STEP2 / Y STEP |
| Y轴 DIR | GPIO34 | DIR2 / Y DIR |
| UART3 TX | GPIO32 | USB-TTL RX |
| UART3 RX | GPIO33 | USB-TTL TX |
| GND | GND | D36A / USB-TTL GND |
| GPIO35 | 不接 | 不接 |

D36A 的 `EN1/EN2` 使用板载 5V 拉高。实测该驱动板 EN 为高电平有效。不要把 D36A 的 5V 接到 K230 GPIO35。

### 1.3 步进换算

```text
电机：1.8°/步
细分：16
每圈脉冲：200 × 16 = 3200 pulse/rev
每个STEP脉冲角度：360 / 3200 = 0.1125°
```

软件软限位：

```text
X_SOFT_LIMIT_STEPS = 3200.0  # ±360°
Y_SOFT_LIMIT_STEPS = 1422.0  # 约 ±160°
```

该限位是软件积分估算，不是编码器反馈。每次开机前需要手动把云台放在机械安全中间位置，然后可用 `[motor,zero]` 清零虚拟步数。

---

## 2. 串口与数据协议

### 2.1 UART

```text
GPIO32 TX → USB-TTL RX
GPIO33 RX ← USB-TTL TX
GND       → USB-TTL GND
波特率：115200
格式：8N1
逻辑：3.3V TTL
```

不要连接 USB-TTL 的 5V/VCC。

### 2.2 plot 数据包

程序连续发送 10 通道数据包：

```text
[plot,err_x,err_y,x_target_hz,y_target_hz,x_output_hz,y_output_hz,target_x,target_y,state,fps]
```

| 通道 | 含义 |
|---:|---|
| 1 | `err_x`，目标中心相对画面中心的X误差，单位px |
| 2 | `err_y`，目标中心相对画面中心的Y误差，单位px |
| 3 | `x_target_hz`，X轴控制器目标STEP频率 |
| 4 | `y_target_hz`，Y轴控制器目标STEP频率 |
| 5 | `x_output_hz`，X轴实际应用STEP频率 |
| 6 | `y_output_hz`，Y轴实际应用STEP频率 |
| 7 | `target_x`，目标中心X坐标 |
| 8 | `target_y`，目标中心Y坐标 |
| 9 | `state`，状态码 |
| 10 | `fps`，循环帧率 |

状态码：

```text
0 = SEARCH / LOST
1 = ACQUIRE
2 = TRACK
3 = COAST
4 = STOP / ESTOP
```

### 2.3 perf 数据包

每约 500ms 发送一次：

```text
[perf,capture_ms,detect_ms,associate_ms,refine_ms,control_ms,display_ms,total_ms,roi_area,full_scan,candidate_count,state]
```

用于判断慢在采集、检测、关联、角点精修、控制还是显示。

---

## 3. 当前最终数据分析

数据文件：

```text
plot_2026-07-14T02-14-00-685Z.csv
```

去重后统计：

| 指标 | 数值 |
|---|---:|
| 原始记录数 | 1031 |
| 去重后记录数 | 837 |
| 有效追踪时长 | 28.93 s |
| TRACK占比 | 91.55% |
| COAST占比 | 8.07% |
| SEARCH/LOST占比 | 0.38% |
| 最长连续LOST | 0.083 s |
| 最长连续COAST | 0.175 s |
| TRACK平均FPS | 72.53 |
| TRACK 10%分位FPS | 66.23 |
| X平均绝对误差 | 22.58 px |
| Y平均绝对误差 | 9.29 px |
| X 90%误差 | 43.00 px |
| Y 90%误差 | 25.00 px |
| X 95%误差 | 54.00 px |
| Y 95%误差 | 36.00 px |
| `target_x` >80px单帧跳变 | 0 次 |
| `target_y` >80px单帧跳变 | 0 次 |
| TRACK目标X范围 | 238~392 |
| TRACK目标Y范围 | 181~308 |

结论：

1. v3.5 已经解决此前“识别到但因软限位不追踪”的主要问题。
2. 抗干扰明显改善，`target_x/y` 没有超过 80px 的单帧跳变。
3. 主要剩余问题是 X 轴动态误差偏大：X MAE 约 22.58 px，Y MAE 约 9.29 px。
4. 电机执行不是瓶颈：`target_hz` 与 `output_hz` 基本一致；问题主要在X轴动态响应参数，不是PWM输出失败。

---

## 4. v3.6 相对 v3.5 的最终修改

v3.6 不再放宽视觉阈值，避免重新引入环境干扰。只针对数据暴露出的 X 轴动态误差加强控制。

| 参数 | v3.5 | v3.6 | 目的 |
|---|---:|---:|---|
| `control_lead_s` | 0.040 | 0.046 | 略增加预测前瞻，减少高速水平滞后 |
| `max_lead_x_px` | 80 | 92 | 允许X轴预测中心稍微更靠前 |
| `x_kp` | 4.80 | 5.15 | 增强X轴比例响应 |
| `x_kd` | 0.105 | 0.125 | 提高X轴阻尼，抑制增强Kp后的冲过中心 |
| `x_ff` | 0.22 | 0.28 | 增强速度前馈 |
| `x_ff_limit_hz` | 240 | 300 | 允许快速水平运动时前馈更充分 |
| `x_near_cap_hz` | 170 | 190 | 中心附近允许更及时修正 |
| `x_mid_cap_hz` | 470 | 540 | 中等误差区域提升水平跟随速度 |

未修改的重点：

- 不放宽 `quality_high / quality_low`。
- 不降低 `min_contrast`。
- 不改变矩形宽高比门限。
- 不提高全局最大频率。
- 不改变Y轴控制，因为本次Y轴误差已经较低。

---

## 5. 视觉算法细节

### 5.1 检测主线

当前使用 `cv_lite` 路线：

```text
RGB888图像
→ 灰度/边缘
→ Canny
→ 轮廓
→ 多边形近似
→ 四边形筛选
→ 候选评分
```

原生 `find_rects()` 在当前 K230 固件上实测全图只有约 11~12 FPS，因此最终版默认关闭原生检测，只保留 `cv_lite` 主线。

### 5.2 候选筛选

候选必须满足：

- 面积范围；
- 最小宽高；
- 宽高比范围；
- 四角近似直角；
- 黑白对比；
- 内外矩形结构；
- 与历史目标面积、宽高比、位置连续。

核心参数：

```python
RECT_CFG = {
    "min_area_detect": 120.0,
    "max_area_ratio": 0.46,
    "min_w_detect": 12,
    "min_h_detect": 9,
    "aspect_min": 1.05,
    "aspect_max": 2.38,
    "target_aspect": 1.50,
    "quality_high": 0.36,
    "quality_low": 0.18,
    "quality_search_recover": 0.30,
    "max_candidates": 18,
    "cv_canny_low": 22,
    "cv_canny_high": 84,
    "cv_approx_epsilon": 0.0205,
    "cv_area_min_ratio": 0.0012,
    "cv_max_angle_cos": 0.40,
    "min_contrast": 12.0,
    "low_stage_min_ring": 0.16,
}
```

### 5.3 动态 ROI

锁定目标后，不再每帧只做全图搜索，而是围绕 Kalman 预测位置生成 ROI：

```python
"roi_scale_w": 3.30
"roi_scale_h": 3.40
"roi_min_w": 150
"roi_min_h": 120
"roi_velocity_lead_s": 0.120
"roi_velocity_margin_gain": 0.16
"roi_extra_margin": 22
"full_scan_interval": 8
"full_scan_after_miss": 1
```

目的：

- 正常跟踪时减少背景候选；
- 快速运动时沿速度方向提前放大 ROI；
- 漏检后尽快全图校验。

### 5.4 两级关联

同一批候选分为：

```text
高置信度候选：quality >= quality_high
低置信度候选：quality_low <= quality < quality_high
```

流程：

```text
高置信度关联
→ 失败
低置信度关联
→ 失败
COAST或SEARCH
```

低置信度候选需要更严格服从历史轨迹，避免被背景矩形接管。

### 5.5 跳变保护

```python
"jump_guard_px": 95.0
"jump_guard_quality": 0.48
"jump_guard_ring": 0.18
"jump_guard_contrast": 12.0
```

如果候选相对上一帧跳变太大，它必须同时具备较高质量、黑白环结构或足够对比度，否则拒绝关联。

---

## 6. 状态机

```text
SEARCH
  找到候选 → ACQUIRE

ACQUIRE
  连续确认目标 → TRACK
  下一帧不一致 → SEARCH

TRACK
  有检测 → 更新Kalman并控制电机
  短时漏检 → COAST
  长时间无检测 → LOST

COAST
  用Kalman预测短时续航
  找回目标 → TRACK
  超过max_coast_frames → LOST

LOST
  停止电机
  优先按历史轨迹重捕获
  长时间无目标 → SEARCH
```

关键参数：

```python
TRACK_CFG = {
    "acquire_frames": 2,
    "reacquire_frames": 2,
    "max_coast_frames": 8,
    "keep_track_ms": 5000,
    "control_lead_s": 0.046,
    "gate_tracking_px": 155.0,
    "gate_reacquire_px": 280.0,
    "jump_guard_px": 95.0,
    "measurement_var_detect": 10.0,
    "measurement_var_recover_scale": 1.20,
    "measurement_var_low_scale": 2.20,
    "max_velocity_x_px_s": 560.0,
    "max_velocity_y_px_s": 500.0,
    "coast_velocity_decay": 0.94,
    "max_lead_x_px": 92.0,
    "max_lead_y_px": 58.0,
}
```

---

## 7. 控制算法

### 7.1 坐标误差

逻辑画面中心：

```python
cx0 = 320
cy0 = 240
```

误差：

```python
err_x = target_x - 320
err_y = target_y - 240
```

正负方向：

- `err_x > 0`：目标在画面右侧，X轴往正方向追；
- `err_x < 0`：目标在画面左侧，X轴往负方向追；
- `err_y > 0`：目标在画面下侧，Y轴往正方向追；
- `err_y < 0`：目标在画面上侧，Y轴往负方向追。

### 7.2 PID + 前馈

控制输出为 STEP 频率：

```text
target_hz = PID(error) + ff_gain × target_velocity
```

PID：

```python
PID_CFG = {
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
```

前馈与限速：

```python
MOTION_CFG = {
    "x_min_hz": 55.0,
    "y_min_hz": 50.0,
    "x_max_hz": 1000.0,
    "y_max_hz": 820.0,

    "x_accel_hz_s": 3600.0,
    "y_accel_hz_s": 3200.0,
    "x_decel_hz_s": 6500.0,
    "y_decel_hz_s": 5800.0,

    "x_ff": 0.28,
    "y_ff": 0.20,
    "x_ff_limit_hz": 300.0,
    "y_ff_limit_hz": 210.0,

    "near_error_px": 18.0,
    "mid_error_px": 55.0,
    "far_error_px": 140.0,
    "x_near_cap_hz": 190.0,
    "y_near_cap_hz": 150.0,
    "x_mid_cap_hz": 540.0,
    "y_mid_cap_hz": 410.0,
    "x_far_cap_hz": 800.0,
    "y_far_cap_hz": 650.0,
}
```

### 7.3 大误差禁止急刹

为避免“误差很大但速度预测穿过中心导致输出为0”，程序限制只有在误差较小区域才允许接近中心制动：

```python
"approach_max_error_px": 68.0
"error_floor_start_px": 24.0
```

当误差大于 `error_floor_start_px` 但目标输出小于最低步频时，会强制给出最低追踪速度。

---

## 8. COAST 短时预测

短时漏检时，程序不会马上停机，而是进入 COAST：

```python
"max_coast_frames": 8,
"coast_velocity_decay": 0.94
```

普通 COAST：

```python
"coast_scale_1": 0.72,
"coast_scale_2": 0.38,
"coast_scale_3": 0.15,
"coast_scale_4": 0.08,
"coast_scale_5": 0.00,
"coast_scale_6": 0.00,
```

目标远离中心时的向外逃逸 COAST：

```python
"coast_outward_error_px": 42.0,
"coast_outward_velocity_px_s": 28.0,
"x_coast_out_max_hz": 720.0,
"y_coast_out_max_hz": 580.0,
```

为了防止无检测时突然暴增速度，COAST 上限还受最后一帧有效 TRACK 指令约束：

```python
"coast_reference_gain": 1.18,
"coast_reference_margin_hz": 45.0,
"coast_edge_reference_margin_hz": 120.0,
```

---

## 9. 常用串口命令

### 9.1 基本控制

```text
[key,start]
[key,stop]
[sv,estop]
[sv,restart]
[motor,zero]
[motor,get_state]
[motor,get_limit]
[system,ping]
```

### 9.2 角度范围

```text
[slider,xLimitDeg,360]
[slider,yLimitDeg,160]
```

更大测试范围：

```text
[slider,xLimitDeg,420]
[slider,yLimitDeg,180]
```

仅在机械和线缆安全时使用。

### 9.3 控制调参

```text
[slider,xKp,5.15]
[slider,xKd,0.125]
[slider,xFF,0.28]
[slider,xMidCap,540]
[slider,xNearCap,190]
[slider,yKp,4.30]
[slider,yKd,0.120]
```

### 9.4 视觉调参

```text
[slider,qHigh,0.36]
[slider,qLow,0.18]
[slider,qSearch,0.30]
[slider,minContrast,12]
[slider,lowRing,0.16]
```

环境干扰严重时先收紧：

```text
[slider,qHigh,0.40]
[slider,qLow,0.22]
[slider,minContrast,14]
```

丢目标严重时再放宽：

```text
[slider,qHigh,0.34]
[slider,qLow,0.16]
[slider,qSearch,0.26]
```

---

## 10. 后续开发注意事项

1. 不要把 `virtual_steps` 当成真实角度反馈；它只是软件积分估算。
2. 每次开机前手动摆到安全中心，再运行 `[motor,zero]`。
3. 看到识别到目标但不转，先看终端 `limit=(x,y)`。
4. `limit=(0,1)` 或 `limit=(1,0)` 表示软件限位拦住了该方向。
5. 若 `state=TRACK` 且 `targetHz`非零但 `outHz=0`，检查软限位或STEP输出。
6. 若 `state=SEARCH`，电机不动是正常行为。
7. 若 `target_x/y` 大跳变，优先收紧视觉阈值，不要先调PID。
8. 若 FPS 低，查看 `[perf,...]` 的 `detect_ms` 和 `display_ms`。
9. 若 X 滞后，优先调 `xFF / xKp / xMidCap`。
10. 若中心抖动，优先加死区或降低近中心速度上限。
11. 若丢失时间长，优先看检测候选数与ROI，不要直接增加COAST帧数。
12. 当前方案不需要KPU模型训练；只有传统视觉无法抗复杂背景时再考虑训练模型。

---

## 11. 在新对话中应提供的文件

建议上传：

```text
main_k230_d36a_rect_track_uart_v3_6_final_rect_tracker.py
K230_D36A_rect_tracker_v3_6_project_background.md
最新一次 plot CSV
如有问题，再附终端 STAT 日志
```

新对话重点说明：

```text
这是K230 + D36A步进云台项目，目标是视觉追踪黑色矩形方框。
当前最终代码为v3.6，使用cv_lite矩形检测、动态ROI、两级关联、Kalman、TRACK/COAST/LOST状态机，以及STEP/DIR频率控制。
硬件引脚固定：GPIO42/26控制X轴STEP/DIR，GPIO43/34控制Y轴STEP/DIR，GPIO32/33为UART3调参串口。
不要改引脚，不要改plot 10通道格式，先分析CSV再调参。
```
