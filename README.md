# K230 + D36A 二维步进云台矩形追踪与激光闭环项目

基于 **K230 / CanMV MicroPython**、**GC2093 摄像头**、**D36A 双路步进驱动板**和二维步进云台的视觉闭环控制项目。

仓库当前主程序是 `v3.6` 矩形追踪基线版：识别黑色矩形框，控制云台使矩形中心靠近摄像头画面中心。项目后续目标是在保留该稳定架构的基础上，引入激光点检测，将控制误差改为：

```text
矩形中心 - 激光点中心
```

从而实现“激光点自动对准矩形框中心”。

---

## 当前状态

| 模块 | 状态 |
|---|---|
| 黑色矩形框检测 | 已实现 |
| `cv_lite` 四边形检测 | 已实现 |
| Kalman 位置与速度估计 | 已实现 |
| 动态 ROI | 已实现 |
| 高/低置信度两级关联 | 已实现 |
| SEARCH / ACQUIRE / TRACK / COAST / LOST 状态机 | 已实现 |
| D36A 双轴 STEP/DIR 控制 | 已实现 |
| PID、速度前馈、加减速 Ramp、换向制动 | 已实现 |
| UART3 在线调参 | 已实现 |
| 10 通道 `plot` 数据输出 | 已实现 |
| 激光 TTL GPIO35 控制 | 当前主分支未启用 |
| 激光点视觉检测 | 待合入主分支 |
| 激光点对准矩形中心闭环 | 开发中 |

> **注意**
>
> 当前仓库文件名虽然包含 `laser_rect_center_track`，但主分支中的 `v3.6` 源码仍是“矩形中心对准摄像头中心”的稳定基线，且 GPIO35 未初始化。不要把当前版本误认为已经完成激光闭环。

---

## 系统架构

### 当前 v3.6 基线

```text
GC2093 摄像头
      │
      ▼
320×240 RGB888 图像
      │
      ▼
cv_lite 四边形检测
      │
      ▼
矩形候选评分与轨迹关联
      │
      ▼
Kalman + 动态 ROI
      │
      ▼
误差 = 矩形中心 - 画面中心
      │
      ▼
PID + 速度前馈 + Ramp
      │
      ▼
D36A STEP/DIR
      │
      ▼
二维步进云台
```

### 目标激光闭环

```text
                    ┌───────────────┐
GC2093 摄像头 ──────┤ 矩形中心检测  │
                    └───────┬───────┘
                            │
                            ▼
                    矩形中心 (xr, yr)
                            │
                            │  err = rect - laser
                            │
                    激光中心 (xl, yl)
                            ▲
                            │
                    ┌───────┴───────┐
GC2093 摄像头 ──────┤ 激光点检测    │
                    └───────────────┘
                            │
                            ▼
                  PID + 前馈 + Ramp
                            │
                            ▼
                      D36A 二维云台
```

---

## 硬件

- K230 开发板
- GC2093 CSI2 摄像头
- D36A 双路步进电机驱动板
- 两相步进电机二维云台
- USB-TTL 模块，3.3 V 逻辑
- 激光笔与 TTL 控制电路，供后续激光闭环版本使用
- 独立电机和激光供电

### 固定引脚

| 功能 | K230 引脚 | 外设连接 |
|---|---:|---|
| X 轴 STEP | GPIO42 | D36A STEP1 |
| X 轴 DIR | GPIO26 | D36A DIR1 |
| Y 轴 STEP | GPIO43 | D36A STEP2 |
| Y 轴 DIR | GPIO34 | D36A DIR2 |
| UART3 TX | GPIO32 | USB-TTL RX |
| UART3 RX | GPIO33 | USB-TTL TX |
| 激光 TTL | GPIO35 | 当前 v3.6 未启用 |
| 公共地 | GND | D36A、USB-TTL、激光控制电路 GND |

### D36A 使能

D36A 的 `EN1`、`EN2` 不由 K230 GPIO 控制，必须接到驱动板自身的 5 V，使驱动保持硬件使能。

**严禁把 D36A 的 5 V 接入任何 K230 GPIO。**

### UART 接线

```text
K230 GPIO32 TX  → USB-TTL RX
K230 GPIO33 RX  ← USB-TTL TX
K230 GND        ↔ USB-TTL GND
```

不要连接 USB-TTL 的 5 V/VCC。

---

## 软件环境

当前代码按以下环境开发和验证：

```text
CanMV v1.4.3
MicroPython e00a144
K230 CanMV IDE
GC2093 CSI2
cv_lite
```

主程序使用：

```python
from machine import PWM, FPIOA, UART, Pin
from media.sensor import *
from media.display import *
from media.media import *
import cv_lite
```

可选的亚像素角点精修依赖：

```python
import cv2
from ulab import numpy as np
```

如果 `cv2` 或 `ulab` 不可用，程序会自动关闭角点精修，不影响主检测链运行。

---

## 仓库结构

```text
.
├── main_k230_d36a_rect_track_uart_v3_6_final_rect_tracker (1).py
└── K230_D36A_rect_tracker_v3_6_project_background (1).md
```

当前文件名保留了本地导出时的 `(1)` 后缀。建议后续整理为：

```text
.
├── main.py
├── README.md
└── docs/
    └── project_background.md
```

---

## 快速开始

### 1. 克隆仓库

普通 HTTPS：

```bash
git clone https://github.com/nini1997420/k230_d36a_laser_rect_center_track.git
```

SSH：

```bash
git clone git@github.com:nini1997420/k230_d36a_laser_rect_center_track.git
```

SSH 443 端口：

```bash
git clone ssh://git@ssh.github.com:443/nini1997420/k230_d36a_laser_rect_center_track.git
```

### 2. 检查机械位置

本系统没有编码器。启动前必须：

1. 关闭电机输出。
2. 手动把二维云台放在机械安全中间位置。
3. 检查电机线、激光线和排线不会被缠绕。
4. 上电后再运行程序。
5. 必要时发送 `[motor,zero]` 清零软件估算位置。

### 3. 在 CanMV IDE 中运行

1. 打开 CanMV IDE。
2. 连接 K230。
3. 打开：

```text
main_k230_d36a_rect_track_uart_v3_6_final_rect_tracker (1).py
```

4. 确认接线和供电。
5. 点击运行。
6. 观察 IDE 图像、终端 `STAT` 日志和 UART 数据。

### 4. 电机自检

程序默认关闭启动自检，避免每次运行自动移动。

通过 UART 发送：

```text
[motor,test]
```

执行 X、Y 两轴正反转短时测试。

---

## 视觉处理

当前检测图像为：

```text
320 × 240 RGB888
```

控制、UART 和历史数据使用：

```text
640 × 480 逻辑坐标
```

主检测流程：

```text
RGB888
→ Canny 边缘
→ 轮廓提取
→ 多边形近似
→ 四边形几何筛选
→ 黑框白心对比度
→ 内外矩形结构评分
→ 轨迹关联
```

### 动态 ROI

目标锁定后，程序围绕 Kalman 预测位置建立动态 ROI，降低背景干扰。漏检或周期检查时恢复全图搜索。

### 两级候选关联

```text
高置信度候选
→ 关联失败
低置信度候选
→ 关联失败
COAST / LOST
```

低置信度候选必须更接近历史轨迹，避免背景矩形接管目标。

---

## 状态机

| 状态 | 含义 |
|---|---|
| `SEARCH` | 全图寻找目标 |
| `ACQUIRE` | 连续确认候选 |
| `TRACK` | 正常检测并控制电机 |
| `COAST` | 短时漏检，使用 Kalman 预测 |
| `LOST` | 目标丢失，停止或等待重捕获 |

UART 状态码：

```text
0 = SEARCH / LOST
1 = ACQUIRE
2 = TRACK
3 = COAST
4 = ESTOP / TRACKING_DISABLED
```

---

## 控制算法

输出不是舵机角度，而是 D36A 的 STEP 频率：

```text
目标频率 = PID(像素误差) + 速度前馈
```

随后经过：

```text
自适应速度上限
→ 接近中心提前制动
→ 最低有效步频补偿
→ 加减速 Ramp
→ 换向先减速到 0
→ PWM STEP 输出
```

默认软限位：

```text
X_SOFT_LIMIT_STEPS = 3200
Y_SOFT_LIMIT_STEPS = 1422
```

按 1.8° 电机、16 细分计算：

```text
3200 pulse/rev
0.1125°/pulse
```

`virtual_steps` 只是按输出频率积分得到的软件估计，不是编码器反馈。堵转、失步或断电后，该值可能与真实位置不一致。

---

## UART 数据协议

默认波特率：

```text
115200 8N1
```

### 10 通道 plot 包

```text
[plot,err_x,err_y,x_target_hz,y_target_hz,x_output_hz,y_output_hz,target_x,target_y,state,fps]
```

| 通道 | 含义 |
|---:|---|
| 1 | X 误差，px |
| 2 | Y 误差，px |
| 3 | X 轴目标 STEP 频率 |
| 4 | Y 轴目标 STEP 频率 |
| 5 | X 轴实际 STEP 频率 |
| 6 | Y 轴实际 STEP 频率 |
| 7 | 目标 X 坐标 |
| 8 | 目标 Y 坐标 |
| 9 | 状态码 |
| 10 | FPS |

当前 v3.6 中：

```text
err_x = 矩形中心X - 画面中心X
err_y = 矩形中心Y - 画面中心Y
```

目标激光闭环版本中应改为：

```text
err_x = 矩形中心X - 激光点X
err_y = 矩形中心Y - 激光点Y
```

### 性能包

```text
[perf,capture_ms,detect_ms,associate_ms,refine_ms,control_ms,display_ms,total_ms,roi_area,full_scan,candidate_count,state]
```

用于区分采集、检测、关联、角点精修、控制和显示耗时。

---

## 常用 UART 命令

### 基本控制

```text
[key,start]
[key,stop]
[sv,estop]
[sv,restart]
[motor,test]
[motor,zero]
[motor,get_state]
[motor,get_limit]
[system,ping]
```

### PID 参数

```text
[slider,xKp,5.15]
[slider,xKi,0]
[slider,xKd,0.125]

[slider,yKp,4.30]
[slider,yKi,0]
[slider,yKd,0.120]
```

### 速度和加速度

```text
[slider,xMinHz,55]
[slider,yMinHz,50]
[slider,xMaxHz,1000]
[slider,yMaxHz,820]

[slider,xAccel,3600]
[slider,yAccel,3200]
[slider,xDecel,6500]
[slider,yDecel,5800]
```

### 视觉门限

```text
[slider,qHigh,0.36]
[slider,qLow,0.18]
[slider,qSearch,0.30]
[slider,minContrast,12]
[slider,lowRing,0.16]
```

### 软限位

```text
[slider,xLimitDeg,360]
[slider,yLimitDeg,160]
```

仅在机械和线缆安全时扩大运动范围。

---

## 调试顺序

不要同时修改视觉和 PID。建议严格按以下顺序排查。

### 1. 确认检测

观察：

```text
state
candidate_count
target_x
target_y
```

目标坐标大幅跳变时，先处理候选筛选和轨迹关联，不要先调 PID。

### 2. 确认输出

若：

```text
state=TRACK
targetHz != 0
outHz = 0
```

检查：

- 软件软限位；
- `limit=(x,y)`；
- D36A EN1/EN2；
- STEP/DIR 接线；
- 电机供电。

### 3. 确认控制方向

目标在右侧时，X 轴应让误差减小；目标在下方时，Y 轴应让误差减小。

机械方向相反时，仅修改：

```python
X_REVERSE
Y_REVERSE
```

不要同时反转误差和方向参数。

### 4. 再调 PID

- 大误差跟不上：优先调整 `Kp`、前馈和中远区速度上限。
- 中心附近振荡：增加死区、降低近区速度或提高阻尼。
- 输出变化快但电机跟不上：降低加速度或最大频率。
- 检测经常丢失：先检查视觉和 ROI，不要盲目增加 COAST 帧数。

---

## 已知限制

1. 没有编码器，无法获得真实云台角度。
2. 软件软限位依赖 STEP 频率积分，失步后会产生位置误差。
3. 当前主分支仍以摄像头中心作为控制参考。
4. GPIO35 激光 TTL 和激光点检测尚未合入当前主分支。
5. 传统视觉对复杂背景、强反光和相似矩形仍可能误检。
6. 参数与机械惯量、细分、电机电流、镜头视场和目标尺寸有关，不能直接适用于所有硬件。
7. 激光具有眼睛和反射伤害风险，调试时不得照射人员、动物、车辆或高反射表面。

---

## 激光闭环开发要求

合入激光闭环时，建议保持当前矩形检测、状态机、UART 和电机架构，只修改控制反馈链，并增加独立的激光检测模块。

核心要求：

```text
矩形检测成功
AND
激光检测成功
→ 启用闭环控制

矩形或激光任一丢失
→ 停止电机或进入受限恢复状态
```

建议使用独立 RGB565 通道和 `find_blobs()` 检测激光点，避免在 MicroPython 中对 320×240 图像执行 Python 双重逐像素循环。

推荐新增的数据字段：

```text
rect_x
rect_y
laser_x
laser_y
rect_minus_laser_x
rect_minus_laser_y
laser_found
```

---

## 基线测试记录

项目背景文档记录的一次 v3.6 前序测试结果：

| 指标 | 结果 |
|---|---:|
| TRACK 占比 | 91.55% |
| COAST 占比 | 8.07% |
| SEARCH/LOST 占比 | 0.38% |
| TRACK 平均 FPS | 72.53 |
| X 平均绝对误差 | 22.58 px |
| Y 平均绝对误差 | 9.29 px |
| 大于 80 px 的单帧目标跳变 | 0 次 |

这些结果仅代表对应硬件、环境和数据文件，不应视为所有场景下的保证。

---

## 后续计划

- [ ] 整理文件名并移除 `(1)` 后缀
- [ ] 将主程序统一命名为 `main.py`
- [ ] 增加 `.gitignore`
- [ ] 增加接线图和实物图片
- [ ] 合入 GPIO35 激光 TTL 控制
- [ ] 增加 RGB565 激光点快速检测
- [ ] 将控制误差改为矩形中心减激光中心
- [ ] 增加激光丢失保护与重捕获
- [ ] 保存可复现的 CSV 测试数据
- [ ] 增加版本变更记录
- [ ] 增加开源许可证

---

## 安全说明

- 激光不得照射眼睛。
- 不要将激光对准人员、动物、车辆、航空器或公共区域。
- 不要在镜子、金属、玻璃等高反射表面附近进行自动追踪。
- 电机调试前确认机械限位和线缆余量。
- D36A 电机电源、K230 和 USB-TTL 必须共地。
- D36A 的 5 V 不得接入 K230 GPIO。
- 改动方向参数后先使用低速点动测试。
