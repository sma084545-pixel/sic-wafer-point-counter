# Rigaku 论文证据与本项目实现对齐

## 证据来源

本说明依据项目目录中保存的两篇原文进行核对：

- *Automated dislocation evaluation software for X-ray topography images - Topography Analysis*, Rigaku Journal 32(1), 2016, pp. 33-35。
- C. Reimann and C. Kranert, *Non-destructive characterization of crystallographic defects of SiC substrates using X-ray topography for R&D and quality assurance in production*, Rigaku Journal 37(2), 2021, pp. 33-37。

参考截图 `205.png` 只用于理解版式和标记语义，不是当前样品的 ground truth，也不纳入公开网站资产。

## 图 5 的标记语义

2021 原文第 35-36 页说明并在图 5 图注中定义：

- 红色矩形：XRT 自动检测的 TSD 位置；
- 黄色圆圈：外延层上由 DIC 显微镜独立检测的表面坑；
- KOH 腐蚀比较被文字提及，但没有在该图展示。

本项目据此新增 `overlay_xrt_red_boxes.png`：红框表示当前规则自动接受的 XRT 点状候选。另输出 `xrt_detection_detail_montage.png`，从同一全局归一化的全分辨率科研灰度通道选取代表性局部视场，以真实候选边界框和物理标尺展示细节。没有导入、配准并核验来源的 DIC/KOH 坐标时，程序不生成黄色圆圈，也不把这些图命名为 XRT-DIC 一致性图。`defect_comparison_details.png` 只是同一 XRT 图像的原始局部与自动判定复核，不是独立物理验证。

## 50 µm × 30 µm 的适用边界

2021 原文第 35 页只在 4H-SiC、Cu Kalpha、(008) 反射条件下，把 TSD 描述为强对比椭圆斑点，近似尺寸 50 µm × 30 µm。论文没有公开可复现的分割阈值或完整分类算法。

项目因此：

1. 在候选 CSV 中保存等效直径、长轴和短轴的像素及微米值；
2. 提供 `config/rigaku_2021_tsd_008.yaml` 作为非门控诊断配置；
3. 只把 50 µm × 30 µm 转为当前像素尺度下的预期像素跨度；
4. 预期短轴少于 3 px 时发出形貌采样不足警告；
5. 不用该尺寸自动改变 `accepted`，也不把所有相似黑点称为 TSD。

只有确认成像几何相容并使用真实专家、DIC、KOH 或其他独立证据验证后，才能建立面向 TSD 的物理分类规则。

## 图 6 的整片密度图

2021 图 6 使用毫米坐标和 `cm^-2` 色条展示两片 150 mm 晶圆。图中的 294、647 和共同 0-1500 cm^-2 色标属于特定样品与比较版式，不是本项目 100 mm 晶圆的目标值或默认色标。

本项目的二维网格对每格计算：

```text
rho_ij = n_ij / S_valid,ij
```

其中 `S_valid,ij` 来自该格内最终 `valid_analysis_mask` 的实际像素面积。输出 `density_heatmap_grid.csv` 保留每格边界、有效像素数、有效面积比例、count、密度和 Garwood Poisson 区间。零有效面积格为 NA。显示层可以隐藏有效面积比例过低的边缘格，并可对色标上限做有记录的百分位截断；CSV 数值不被修改。

网格边界会扩展到覆盖所有真实有效像素，避免拟合圆残差导致面积遗漏。摘要记录热图面积与主有效面积的相对误差、count 是否等于接受数、网格尺寸、色表和显示截断状态。

## 采样与径向加权

2021 原文报告了特定 150 mm 晶圆上全片、13 个 15 mm 网格场和直径条带方案的偏差，并指出未加权条带过度代表晶圆中心。论文没有公开可直接复用的径向权重公式。

当前程序分析完整晶圆有效掩膜，不把论文的 4.9%、6.6% 或其他采样偏差作为本项目不确定度。若未来只分析局部网格或条带，必须根据实际采样掩膜和整片有效面积推导权重，并使用独立晶圆验证。

## 当前能够与不能够得出的结论

当前能够报告：

- 满足当前图像和形态规则的点状 XRT 候选数 `n`；
- 最终有效面积 `S`；
- `rho = n/S`；
- 逐候选形态、灰度、坐标与拒绝原因；
- 面积归一化的整片、径向、角向和分区密度；
- 只包含有限计数波动的 Poisson 区间。

当前不能据论文或参考截图声称：

- 每个自动候选都是 TSD；
- 当前结果已与 DIC 或 KOH 达到论文所述一致性；
- 当前真实数据具有某个 precision、recall 或 F1；
- 50 µm × 30 µm 是所有 XRT 条件下的固定阈值；
- 长线一定是 BPD、弱点一定是 TED；
- 论文样品的密度或色标应在当前样品上复现。
