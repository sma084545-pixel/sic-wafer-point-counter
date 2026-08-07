# 真实 SiC 专家标注与验证协议

使用 `scripts/validate_real_annotations.py --output validation/` 可先生成 `annotation_template.csv`。每一条原始标注必须保留以下版本化字段：`image_id`、`wafer_id`、`candidate_id`、`x_px`、`y_px`、`x_mm`、`y_mm`、`label`、`reviewer_id`、`review_confidence`、`notes`、`source_image_sha256`、`annotation_schema_version`。

`confirmed_point` 是明确正类。`line_fragment`、`scratch`、`particle_or_dust`、`detector_artifact` 与 `large_dark_region` 是明确负类。`possible_point` 和 `uncertain` 不是负类，必须单独保留；它们不能被静默计为误检。

至少两位标注者时，程序保留全部原始决定，输出原始一致率、适用时的 Cohen's kappa、逐类一致率和分歧表；仲裁标签保存为独立文件，绝不覆盖原始标注。一位标注者时报告 `inter_reviewer_agreement_not_available`。

所有校准、验证与锁定测试拆分以 `wafer_id` 为最小单位。同一晶圆的不同 patch 不能跨集合。锁定测试晶圆不得用于调阈值、选择特征或选择最优 F1。参数敏感性分析只能在校准晶圆上对少量关键参数做预先规定的 ±20% 局部扰动。

验证匹配优先使用毫米坐标和 `matching_tolerance_um`。报告 TP、FP、FN、precision、recall、F1、每 cm² 的 FP/FN、自动与人工的 n/rho 差值以及 Bland–Altman 表。若标注覆盖并不完整，未匹配自动目标标为不确定，precision/F1 不应作为完整性能声明。

标注中的 `source_image_sha256` 在原图仍可访问时会自动核验；哈希不一致或同一图存在互相冲突的声明哈希会停止验证。原图已归档或不可访问时报告 `source_file_unavailable_for_verification`，不能把它当作已核验。

不确定度应分列：泊松计数区间、按晶圆 bootstrap 的分类偏差、参数敏感性、面积/标定不确定度（仅在用户提供输入时）和空间异质性。它们默认不应合并成一个单一 ± 值。
