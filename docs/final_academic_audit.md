# 最终学术审计

审计日期：2026-07-13。代码版本：发布提交前工作树。完整 pytest：**40 passed**。

## 数据链核查

1. `n` 是按当前图像形态与强度规则接受的黑色点状目标数；不是自动宣称的真实位错数。
2. `S` 是 `valid_analysis_mask` 的有效像素数乘以 `(cm_per_pixel)^2`。它不以理想 78.5398 cm² 替代实际面积。
3. 晶圆轮廓外、配置的 edge exclusion、invalid regions/invalid mask 都不进入 `S`，也不应作为接受目标。
4. 可选 `*_um`/`*_um2` 参数在一次运行中依据 `um_per_pixel` 统一转换；物理值优先于旧 `*_px` 值，且写入 `resolved_physical_parameters.yaml`。
5. uint8、uint16 和浮点原始图以同一个全局百分位窗口归一化为 float32 `[0,1]` 后进入预处理与检测；预览/PNG 才量化为 uint8。
6. 全图和 tile 读取使用 ImageSource 的同一 low/high 窗口。WhiteIsZero 仅在这一科学归一化边界反相一次。
7. 当前没有真实 SiC 专家标注验证；主报告状态为 `not validated on real SiC data`。现有 precision/recall/F1 仅来自合成数据测试。
8. 真实标注工具按 `wafer_id` 强制 calibration、validation 与 locked test 互斥；锁定测试不传入参数敏感性 evaluator。
9. 泊松区间仅包含有限计数随机误差。分类 bootstrap、参数敏感性、面积标定不确定度和空间异质性均分列；没有真实输入时明确为未量化或描述性。
10. `radial_density.csv`、`angular_density.csv` 与 `regional_density.csv` 逐 bin 使用最终有效掩膜面积。角度参考固定为 `image_positive_x`，不解释为晶向。
11. 自动几何低可信度、明显裁剪且未手动提供中心/半径时，晶圆检测抛出错误并拒绝输出 rho。
12. HTML 与 JSON 均使用“点状目标”表述，未发现把黑点无证据升级为真实位错的表述。

## 端到端审计样本

| 样本 | n | S (cm²) | rho (cm^-2) | 关键核查 |
|---|---:|---:|---:|---|
| clean 合成晶圆 | 96 | 78.700102 | 1.219821 | 合成真值 96；float32 分析路径 |
| uint16 低动态范围晶圆 | 96 | 78.700102 | 1.219821 | 保持 float32，未量化为 uint8 |
| edge exclusion + invalid/notch 区域 | 96 | 75.063617 | 1.278915 | 实际有效面积下降并重新计算 rho |

三个样本的径向、角向和区域有效面积与 `valid_analysis_area_cm2` 的差均在浮点舍入量级（约 `1e-14 cm²`）。

## 问题分级

### BLOCKER

无。当前代码路径满足合成与 I/O/面积/单位测试，且几何失败会拒绝密度输出。

### MAJOR

无未修复代码错误。审计中发现两个问题并已修复：

- lazy tile 的边界距离可能受 tile 边缘截断；现改为仅对含候选的 mask-only 区域逐步扩展，直到包含真实最终有效边界。
- 真实轮廓略超出拟合圆时，原径向/区域 bin 可遗漏少量有效面积；现以最终有效掩膜的最外有效半径确定空间分箱外界，并通过面积守恒测试。

### MINOR / 外部数据限制

- 尚无材料专家提供的真实 SiC 标注，因此不能报告真实 precision、recall、F1、分类 bias 或论文级定量准确率。
- 尚未提供晶圆实际直径误差、像素标定误差和 invalid-mask 边界误差，面积/标定不确定度只能标记为 `not quantified`。
- 单片晶圆的空间差异是描述性指标；机制结论需要多晶圆和实验/生长元数据。
