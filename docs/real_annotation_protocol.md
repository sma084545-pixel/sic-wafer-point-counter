# 真实 SiC 专家标注与验证协议

使用 `scripts/validate_real_annotations.py --output validation/` 可先生成 `annotation_template.csv`。每一条原始标注必须保留以下版本化字段：`image_id`、`wafer_id`、`candidate_id`、`x_px`、`y_px`、`x_mm`、`y_mm`、`label`、`reviewer_id`、`review_confidence`、`notes`、`source_image_sha256`、`annotation_schema_version`。

`confirmed_point` 是明确正类。`line_fragment`、`scratch`、`particle_or_dust`、`detector_artifact` 与 `large_dark_region` 是明确负类。`possible_point` 和 `uncertain` 不是负类，必须单独保留；它们不能被静默计为误检。

至少两位标注者时，程序保留全部原始决定，输出原始一致率、适用时的 Cohen's kappa、逐类一致率和分歧表；仲裁标签保存为独立文件，绝不覆盖原始标注。一位标注者时报告 `inter_reviewer_agreement_not_available`。

所有校准、验证与锁定测试拆分以 `wafer_id` 为最小单位。同一晶圆的不同 patch 不能跨集合。锁定测试晶圆不得用于调阈值、选择特征或选择最优 F1。参数敏感性分析只能在校准晶圆上对少量关键参数做预先规定的 ±20% 局部扰动。

## 本机候选训练窗口

本机工作台结果页允许对程序已经提出的候选标记 `target`、`artifact` 或 `uncertain`，并选择 `calibration`、`validation` 或 `locked_test`。原始记录写入 `training/candidate_annotations.csv`，同时快照保存候选特征、结果文件 SHA-256、标注者、拆分和时间；模型写入 `training/candidate_classifier.json`。同一候选出现多标注者冲突、任何不确定标签或拆分冲突时，不进入训练。

候选模型使用物理尺度下的尺寸、形态和对比度特征，不使用候选在晶圆上的坐标，避免把采样位置误学成目标身份。至少需要 5 个目标和 5 个伪影校准标签；同一 `wafer_id` 跨集合会停止训练。概率大于等于接受阈值才计入 `n`，小于等于拒绝阈值判为伪影，中间区间明确列为不确定并排除。`outside_valid_mask` 和 `near_wafer_edge` 始终属于硬拒绝，模型不能覆盖。

训练拟合指标不能当作真实准确率。只有独立 `validation` 或 `locked_test` 同时含正负类时，模型文件才附带留出指标；这些指标仍只验证“专家图像候选标签”，不构成 TSD、TED 或 BPD 的物理身份确认。模型 JSON 带内容 SHA-256，可下载后导入 GitHub Pages 浏览器分析页复用；每次应用都会把模型哈希、阈值、概率、规则判定和最终判定写入结果。

## 原图像素交互训练

像素项目保存原图 SHA-256、wafer_id、split、标注者、ROI 原始像素坐标、无损标签 RLE、类别定义、特征配置、固定种子、训练参数、模型、验证状态和操作历史。目标、背景和 ignore 必须分开；ignore 不能被当作负类。训练项目 JSON 不嵌入原始像素，重新打开时必须核对同一原图哈希。

按晶圆防泄漏同时检查 `wafer_id` 与原图 SHA-256。calibration 才参与拟合；validation 和 locked_test 只评分。报告应分别提供像素 precision/recall/F1/IoU、目标 precision/recall/F1、FP/FN 和定位误差。当前 ROI 的训练表现只用于交互纠错；没有独立真实 SiC 标签时，验证状态必须保持“尚不能证明真实准确率提升”。

验证匹配优先使用毫米坐标和 `matching_tolerance_um`。报告 TP、FP、FN、precision、recall、F1、每 cm² 的 FP/FN、自动与人工的 n/rho 差值以及 Bland–Altman 表。若标注覆盖并不完整，未匹配自动目标标为不确定，precision/F1 不应作为完整性能声明。

标注中的 `source_image_sha256` 在原图仍可访问时会自动核验；哈希不一致或同一图存在互相冲突的声明哈希会停止验证。原图已归档或不可访问时报告 `source_file_unavailable_for_verification`，不能把它当作已核验。

不确定度应分列：泊松计数区间、按晶圆 bootstrap 的分类偏差、参数敏感性、面积/标定不确定度（仅在用户提供输入时）和空间异质性。它们默认不应合并成一个单一 ± 值。
