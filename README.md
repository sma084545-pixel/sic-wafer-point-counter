# SiC 晶圆 X 射线形貌图点状目标计数器

这是一个可运行、可调参、可复核的传统计算机视觉程序，用来统计 4 英寸（本项目按直径 **100 mm** 标定）SiC 晶圆图像中的黑色点状目标，并按实际有效分析面积计算点状目标密度。程序不使用深度学习；主要依赖 OpenCV、NumPy、SciPy、scikit-image 和 tifffile。

> **科学解释边界**：程序统计的是“满足当前图像判定和筛选标准的黑色点状目标”。黑点并不会自动成为物理意义上的位错。只有在人工标注或独立实验已确认这类图像特征与位错一一对应时，`n/S` 才可以作为位错密度报告。本工具不能替代材料专家的物理判定。

## 在线静态展示

项目的公开展示页位于 <https://sma084545-pixel.github.io/sic-wafer-point-counter/>，其中[浏览器分析页](https://sma084545-pixel.github.io/sic-wafer-point-counter/analyze.html)会在独立 Web Worker 内运行打包后的同一套 Python 管线。输入 `File` 只读挂载在当前标签页，不上传到网站服务器；页面按格式和内存风险分层放行，兼容布局的 TIFF/BigTIFF 可选择至 100 MiB 并按原始分辨率重叠分块分析。超过网页安全清单、编码不支持、需要完整候选裁剪或需长期留存结果时，应使用本机工作台。

## 程序做什么

一轮分析依次完成：

1. 读取 PNG、JPG、BMP、TIFF/BigTIFF，保留原始数据类型信息，并用可配置分位数把 8 位、16 位或浮点图像归一化；
2. 从低分辨率预览图分割晶圆轮廓，拟合圆心和半径，报告圆度、拟合残差、边界裁剪情况及可信度；
3. 根据拟合圆、边缘排除宽度和无效区生成 `full_wafer_mask` 与 `valid_analysis_mask`；
4. 通过大尺度背景校正和多尺度黑帽/DoG 增强暗色点；
5. 阈值、形态学清理、连通域标记和可选的保守分水岭生成候选；
6. 提取每个候选的位置、尺寸、形状、对比度和边缘距离，拒绝明显噪点、长线、大斑、外部/边缘目标；
7. 保存全部候选、接受项、拒绝项、拒绝原因、局部裁剪和带编号叠加图；
8. 按最终有效掩膜面积计算 `rho = n / S`、计数标准不确定度和 Poisson 95% 区间。

自动晶圆识别低于配置的可信度下限时，分析会明确报警并停止密度计算；不会用一个明显错误的圆继续给出看似精确的结果。此时可以用 `--center-x`、`--center-y` 和 `--radius-px` 提供人工几何标定。

## 项目结构

```text
sic_wafer_point_counter/
├── README.md
├── pyproject.toml
├── requirements.txt
├── config/
│   └── default.yaml
├── docs/
│   ├── academic_baseline.md
│   ├── final_academic_audit.md
│   ├── measurement_protocol.md
│   ├── real_annotation_protocol.md
│   ├── showcase_platform.md
│   ├── index.html                 # GitHub Pages 静态科研展示页
│   ├── assets/showcase.css
│   └── showcase_design/          # 概念、设计系统、实现截图与差异清单
├── src/sic_wafer_counter/
│   ├── __init__.py
│   ├── cli.py
│   ├── pipeline.py
│   ├── image_io.py
│   ├── wafer_detection.py
│   ├── preprocessing.py
│   ├── point_detection.py
│   ├── feature_extraction.py
│   ├── physical_parameters.py
│   ├── density.py
│   ├── visualization.py
│   ├── reporting.py
│   ├── validation.py
│   ├── run_repository.py        # 安全的历史结果与候选分页读取
│   ├── web.py                   # 本机 Flask 工作台与 API
│   ├── utils.py
│   ├── resources/default.yaml   # 安装后的内置默认配置
│   ├── templates/index.html
│   └── static/                  # 原生 CSS 与按职责拆分的 JavaScript
├── scripts/
│   ├── generate_synthetic_wafer.py
│   ├── review_results.py
│   ├── validate_real_annotations.py
│   ├── bootstrap_local_web_workbench.py
│   ├── run_local_web_workbench.sh
│   └── install_macos_web_workbench.py
├── tests/
│   ├── conftest.py
│   ├── test_density.py
│   ├── test_image_io_precision.py
│   ├── test_physical_units.py
│   ├── test_wafer_detection.py
│   ├── test_point_detection.py
│   ├── test_spatial_density.py
│   ├── test_synthetic_pipeline.py
│   ├── test_validation.py
│   ├── test_run_repository.py
│   └── test_web_workbench.py
├── sample_data/README.md
└── 启动SiC晶圆浏览器工作台.command
```

## 安装

要求 Python 3.10 或更新版本。macOS 自带的 Python 3.9 不能运行本项目；不要仅因为命令名是 `python3` 就假定版本满足要求，应先运行 `python3 --version` 检查。

macOS 新用户解压 GitHub ZIP 后，可以直接双击项目根目录的 `启动SiC晶圆浏览器工作台.command`。启动器本身可由系统 Python 3.9 执行，但只负责寻找 `python3.10` 至 `python3.14`、创建隔离环境并安装依赖；科研程序始终只在 Python 3.10+ 中运行。若已有 `.venv` 来自 Python 3.9，启动器会保留它并改建 `.venv-sic`，不会覆盖或伪装成功。第一次安装和构建 Matplotlib 字体缓存可能需要数分钟，启动器会等待服务真正监听后才打开页面。

手工安装时，在项目根目录执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

也可以安装固定的核心依赖清单：

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`pyvips` 是可选的大图后端，不安装也不会影响普通图或可内存映射 TIFF：

```bash
python -m pip install -e ".[large-image]"
```

注意，Python 的 `pyvips` 包还需要操作系统中的 `libvips`。如果它不可用，程序会回退到 tifffile/OpenCV，并把是否真正随机分块读取以及任何内存限制写入日志和摘要。

## 快速演示

先生成三组固定随机种子的模拟图及独立真值表：

```bash
python scripts/generate_synthetic_wafer.py --all --output-dir sample_data/generated
```

完整分析一张 clean 模拟图：

```bash
python -m sic_wafer_counter.cli analyze \
  sample_data/generated/synthetic_clean.png \
  --output results/synthetic_clean \
  --wafer-diameter-mm 100 \
  --config config/default.yaml \
  --save-intermediates \
  --verbose
```

查看所有 CLI 参数：

```bash
python -m sic_wafer_counter.cli analyze --help
```

## 本机展示与分析平台

公开 GitHub Pages 另提供受内存上限保护的[浏览器分析页](https://sma084545-pixel.github.io/sic-wafer-point-counter/analyze.html)：输入 `File` 由 WORKERFS 只读挂载到 Pyodide Worker，不在页面主线程整体读取，也不上传到服务器；分析调用项目打包的同一个 `sic_wafer_counter.pipeline.analyze_image`，不是 JavaScript 仿制算法。它会生成核心指标、全部候选 CSV、论文语义一致的自动 XRT 候选红框图、明确标为非 DIC/KOH 验证的局部复核图、按真实有效面积归一化的整片密度热图及逐格 CSV、HTML 报告和完整 ZIP。

网页限制是分层的，而不是只看压缩文件大小：

- 所有格式的选择上限为 100 MiB；
- PNG/JPG/BMP 仍走整图解码，限制为 24 MiB、600 万像素、单边 6000 px；
- TIFF/BigTIFF 在二维单通道、axes/尺寸合法且有界随机访问后端可用时，可到 100 MiB、1.2 亿像素、单边 16000 px，并固定使用 1024 px tile 与 128 px halo；
- TIFF 分段解码可能超过安全上限、存在未支持的多页/多通道维度，或运行时无法确认有界读取时，网页会停止且不显示 `rho`，不会退回整图解码或缩图检测；
- 为避免结果尾声重复占用内存，较大的候选 CSV 只放入完整 ZIP，不再同时复制为网页单项文件；浏览器模式也不输出每个候选的独立原始位深裁剪。

版本 0.2.1 的发布验收使用 Chrome 151 实际选择并完成了一张 93.5 MiB、7000×7000、uint16、二维单通道未压缩 TIFF：网页与本机均识别 96 个合成真值点，得到相同的 `S=78.539295 cm²` 和 `rho=1.222318 cm^-2`；网页摘要确认 `WORKERFS_read_only_File`、有界区域读取、45 个 1024 px tile、float32 科研通道、`analysis_downsample_factor=1`，完整 ZIP 通过完整性检查。该验收只证明这种文件布局和浏览器路径可运行，不代表所有约 100 MiB 的 TIFF 编码都受支持，也不构成真实 SiC 识别准确率。

Pyodide 和科学计算依赖从固定版本的 jsDelivr 资源下载；输入图像不会发送到该 CDN。完整候选裁剪、无网环境、网页拒绝的 TIFF 编码或需要长期保留结果时，请使用下面的本机工作台。关闭或刷新浏览器标签页前应先下载 ZIP，因为浏览器临时文件不会持久保存。

安装依赖后，启动本机页面：

```bash
.venv/bin/python -m sic_wafer_counter.cli web --workspace .
```

在浏览器打开终端显示的 `http://127.0.0.1:8765/`。页面提供总览、新建分析、历史结果、结果仪表板、候选浏览器，以及方法/验证说明。上传和合成演示都直接调用与 CLI 相同的 `analyze_image` 管线；前端不复制检测算法，也不读取 ground truth 生成结果。

平台的关键行为：

- 后台固定单任务队列，状态只有 `queued`、`running`、`completed`、`failed`，未知耗时使用不确定进度条；
- 服务重启后从 `results/` 的合法直接子目录恢复历史结果；损坏摘要被隔离而不影响其他记录；
- `defects_all.csv` 在服务端流式筛选和分页，单页最多 200 条，不把 10 万候选一次送入浏览器；
- 叠加图、掩膜、空间密度图和报告只在本次运行目录内按白名单提供，缺失文件显示“本次运行未生成”；
- 合成演示醒目标为“合成”，clean/noisy/difficult 都重新经过真实管线，不把合成性能描述成真实 SiC 准确率；
- 失败运行和低可信度晶圆检测保留摘要与日志，并明确拒绝显示 `rho`。

页面只监听本机回环地址，不会将图像上传到第三方服务，不加载 CDN、远程字体或跟踪脚本。不要把开发服务器改为 `0.0.0.0`，也不要通过公网隧道暴露真实样品和候选裁剪。

在 macOS 上，推荐双击项目根目录的 `启动SiC晶圆浏览器工作台.command`。第一次也可运行下列命令安装本机后台服务；这样即使关闭终端，页面也会保持可访问，并在登录后自动恢复：

```bash
.venv/bin/python scripts/install_macos_web_workbench.py
```

该服务仍只监听 `http://127.0.0.1:8765`，不使用或暴露公网域名。若需移除它：

```bash
.venv/bin/python scripts/install_macos_web_workbench.py --uninstall
```

后台服务标签为 `org.sic-wafer-counter.local-workbench`，采用 `RunAtLoad` 与 `KeepAlive`：关闭终端不会停止服务，重新登录后会自动恢复。日志位于 `results/web_workbench_launchd.log`。安装器只写当前用户的 `~/Library/LaunchAgents`，不会创建公网服务。

### 浏览器限制与常见故障

- 已使用桌面 Chrome/Chromium 验证；Safari、Firefox 等当前版本使用的均为标准 HTML/CSS/JavaScript，但本轮未逐浏览器截图验收。
- 页面显示 `ERR_CONNECTION_REFUSED` 表示本机 8765 端口没有服务监听，不是“域名错误”。先双击 `启动SiC晶圆浏览器工作台.command`，或运行 `./scripts/run_local_web_workbench.sh --open`；仍失败时检查 `results/web_workbench_server.log` 或 launchd 日志。
- 双击启动器会通过本机 `/api/health` 同时核对软件版本和当前项目目录；如果 8765 正由旧版或其他程序占用，它不会误开旧页面，也不会终止不明进程，而会自动尝试 8766–8785 并打开匹配当前下载包的地址。
- 超过上传上限、格式不支持或手工圆心/半径只填写一部分时，页面会在表单附近显示可操作错误；服务端仍会重复验证。
- `results/` 和 `candidate_crops/` 可能达到数 GB。平台不会自动删除科研结果，需由研究人员在另行备份后管理磁盘。
- 浏览器只显示已有 PNG 预览，不会为缺失缩略图临时读取原始 TIFF；完整定量值以 CSV/JSON/HTML 报告为准。

平台架构、API、安全模型和展示验收记录见 [展示平台说明](docs/showcase_platform.md)。

典型真实 TIFF 命令：

```bash
python -m sic_wafer_counter.cli analyze \
  "/path/to/SiC_wafer.tif" \
  --output results/sample_001 \
  --config config/default.yaml \
  --wafer-diameter-mm 100 \
  --exclude-edge-mm 1.0
```

`--exclude-edge-mm` 的默认值在 `config/default.yaml` 中明确为 **0 mm**；程序不会偷偷丢掉边缘数据。需要禁用分水岭时加 `--no-watershed`，临时切换阈值可用 `--threshold-method otsu|adaptive|quantile`。

## 如何准备图像

- 最理想的输入包含整片晶圆以及少量外围背景，圆周尽量不被画框、标题或界面元素覆盖。
- 不要先把 16 位 TIFF 截成 8 位。程序按 `io.normalization_low_percentile` 和 `high_percentile`（默认 1%/99%）做稳健归一化，并记录原始 dtype、最小/最大值和归一化窗口。
- 建议保留原始 TIFF，不要用有损 JPG 保存科研原图。
- 如果一张 PNG 只是晶圆内部的局部裁剪、图中没有圆周，也没有独立的像素标尺，则仅凭这张图不能可靠推断 100 mm 晶圆的 `mm_per_pixel`，因此不能报告整片晶圆意义上的绝对面积密度。可以检测局部点，但必须另行提供标定与有效区域。
- 如果有平边、定位缺口、遮挡、裁剪、文字覆盖或坏区，应让它们进入无效区域掩膜；不能仍把理论圆面积当作有效面积。

## 晶圆标定、面积与坐标

100 mm 等于 10 cm。完整理想圆片理论面积为：

```text
S_total = pi × (5 cm)^2 = 78.539816... cm^2
```

拟合到原图的晶圆直径为 `D_px` 时：

```text
mm_per_pixel = 100 / D_px
cm_per_pixel = 10 / D_px
A_pixel = (cm_per_pixel)^2
S_valid = count_nonzero(valid_analysis_mask) × A_pixel
rho = n / S_valid                 [cm^-2]
sigma_rho = sqrt(n) / S_valid     [cm^-2]
```

实际计算必须使用最后一式中的像素掩膜面积。`78.5398 cm²` 只能描述没有缺口、平边、遮挡、裁剪、坏区或边缘排除的理想完整圆；它不能无条件替代 `S_valid`。报告同时区分理论面积、拟合圆面积、边缘排除面积、其他无效面积和最终有效面积。

目标位置以晶圆中心为原点：

```text
x_mm = (x_px - center_x_px) × mm_per_pixel
y_mm = -(y_px - center_y_px) × mm_per_pixel
```

图像的 y 轴向下，而物理坐标 y 轴向上，所以第二式带负号。

若自动圆拟合失败，可手动给出原图坐标（不是预览图坐标）：

```bash
python -m sic_wafer_counter.cli analyze input.tif \
  --output results/manual_geometry \
  --center-x 24120 --center-y 23890 --radius-px 22150 \
  --wafer-diameter-mm 100 --config config/default.yaml
```

建议先打开输出预览核对圆周。错误的半径会同时改变有效像素数和像素物理面积，是密度结果中非常重要的系统误差来源。

## 算法与可调参数

所有阈值都保存在 YAML，不把任何圆度或尺寸阈值写成不可见的物理常数。默认配置的主要部分是：

```yaml
wafer:
  diameter_mm: 100.0
  exclude_edge_mm: 0.0

io:
  preview_max_size: 2000
  tile_size: 2048
  tile_overlap: 128
  normalization_low_percentile: 1.0
  normalization_high_percentile: 99.0

preprocessing:
  median_kernel: 3
  gaussian_sigma: 1.0
  background_method: morphological
  background_kernel_px: 101
  use_clahe: false

detection:
  method: blackhat       # blackhat、dog 或 combine
  threshold_method: otsu
  use_watershed: true
  min_peak_distance_px: 5

filters:
  min_area_px: 5
  max_area_px: 1500
  max_equivalent_diameter_px: 45.0
  min_circularity: 0.25
  max_eccentricity: 0.92
  max_aspect_ratio: 4.0
  min_solidity: 0.60
  min_contrast: 0.02
```

### 物理单位参数（推荐用于跨分辨率比较）

上述 `*_px` 是兼容旧配置的像素参数。若同一研究要比较不同像素分辨率的图像，可在 YAML 中改用物理参数：`background_kernel_um`、`gaussian_sigma_um`、`blackhat_kernel_sizes_um`、`min_peak_distance_um`、`dog_min_sigma_um`、`dog_max_sigma_um`、`min/max_equivalent_diameter_um`、`min/max_area_um2`、`local_background_ring_um` 和 `min_edge_distance_um`。程序以拟合晶圆的 `um_per_pixel = mm_per_pixel × 1000` 统一换算；物理参数与旧像素参数同时存在时，物理参数优先，并在警告和 `resolved_physical_parameters.yaml` 中记录。形态学核会被转换为正奇数像素。

2021 年 Rigaku 论文在 **4H-SiC、Cu Kalpha、(008) 反射**条件下报告 TSD 图像特征约为 50 µm × 30 µm。项目把它实现为默认关闭、且绝不参与自动接受/拒绝的文献诊断档案；可直接使用完整配置 `config/rigaku_2021_tsd_008.yaml` 查看该尺寸在当前像素尺度下对应多少像素。只有确认成像条件相容后才可把 `imaging_conditions_confirmed` 改为 `true`。若预期短轴少于 3 px，程序会警告形貌采样不足。该尺寸不能跨衍射几何硬编码为通用 TSD 阈值。原文证据与实现边界见 [Rigaku 文献对齐说明](docs/rigaku_2021_alignment.md)。

论文图 5 中的黄色圆圈不是算法自动生成的第二种候选，而是独立 DIC 坑位。若要得到同语义的红框/黄圈对照，先建立带来源哈希的模板：

```bash
python scripts/create_registered_reference_template.py \
  input_xrt.tif independent_dic.tif \
  --method DIC \
  --output registered_dic_points.csv
```

完成图像配准后填写坐标、标签和配准误差，再在配置的 `independent_reference` 中启用 CSV 与参考图。只有 `confirmed_point` 且 `registration_status=registered`、源 XRT 与独立参考图 SHA-256 均匹配的记录才会画成黄色圆圈。`possible_point` 和 `uncertain` 只保留在审计表中；任何来源或坐标校验失败都会停止报告，参考点永远不会改变自动 `n` 或 `rho`。若独立参考并非覆盖完整视场，程序不会把未匹配自动候选直接算作假阳性，也不会计算 precision/F1。

候选 CSV 同时保存 `distance_to_fitted_circle_mm`（旧的圆拟合距离）与 `distance_to_valid_boundary_mm`。后者来自最终 `valid_analysis_mask` 的欧氏距离变换，因此会计入平边、notch、边缘排除带和无效区域；自动的 `near_wafer_edge` 筛选使用后者。旧列 `distance_to_wafer_edge_mm` 为兼容既有表格而保留，语义等同于拟合圆距离。

预处理不会覆盖原图。中值/高斯滤波只做轻度去噪；大尺度闭运算或高斯背景估计产生 `background`，暗目标响应为 `max(background - image, 0)`。背景核应明显大于典型黑点，过小会削弱目标或制造环状响应。CLAHE 和一维条纹抑制默认关闭，启用后也应通过标注数据验证。

候选检测支持两条路径：

- `blackhat`：多尺度形态学黑帽响应加 Otsu、自适应或分位数阈值；
- `dog`：Difference of Gaussian 斑点检测；
- `combine`：按配置求并集或交集。

候选经过开/闭运算和连通域标记。默认 `otsu_classes: 3` 的多类 Otsu 会把背景、普通暗点和极暗大伪影分开，并使用第一条分界；设为 `2` 可恢复经典二类 Otsu。可选分水岭使用距离变换局部极大值分开粘连点；程序有意让高长宽比、高离心率或低圆度的原始连通区绕过分水岭，避免把划痕、文字沿长度切成许多“点”。报告保留分水岭前后数量，数量突增是需要人工检查的过分割信号。

每个候选至少记录：编号、像素/毫米坐标、径向距离和方位角、面积、周长、等效直径、长短轴（像素及微米）、长宽比、离心率、圆度、solidity、边界框、原始灰度、暗响应、局部背景、对比度、距晶圆边缘距离、`accepted` 和 `rejection_reason`。拒绝原因可能包括：

```text
too_small, too_large, too_elongated, low_circularity,
low_solidity, low_contrast, near_wafer_edge, outside_valid_mask
```

筛选同时使用多个形态与强度特征，而不是把所有黑像素、或单一圆度阈值，直接当成位错。

## 参数校准：默认值不是普适物理标准

`default.yaml` 只是软件起点，**不能把默认阈值视为所有 SiC 材料、成像条件、衍射条件或曝光参数下通用的物理标准**。接入真实数据时建议：

1. 由材料研究人员在有代表性的晶圆、中心/边缘区域和不同信噪比图像上逐点标注；
2. 明确哪些图像目标可当作研究定义下的“真阳性”，并单独标注划痕、灰尘、文字、标尺、大斑和坏点；
3. 把图像分成校准集与独立验证集，不要在同一批图上调参和报最终性能；
4. 在校准集上扫描面积、尺寸、圆度、离心率、solidity 和对比度阈值；
5. 在验证集上报告 precision、recall、F1，并按晶圆径向区间检查漏检是否有偏；
6. 保存所用配置和软件版本，参数改变后重新验证。

阈值校准的目标应由研究用途决定：若漏检代价更高，可以提高 recall 后加强人工复核；若假阳性代价更高，则提高 precision，但要诚实报告漏检风险。

真实 SiC 标注验证使用 `scripts/validate_real_annotations.py`；未传入标注时，它只会生成版本化 CSV 模板和 `not validated on real SiC data` 状态，不会伪造真实 precision、recall、F1 或分类不确定度。标注格式、双标注者仲裁、按 `wafer_id` 拆分 calibration/validation/locked test 与不确定度边界见 [真实标注协议](docs/real_annotation_protocol.md)。

空间分布输出使用最终有效掩膜的实际面积，而不是理想圆环面积。二维热图同样计算 `rho_ij = n_ij / S_valid,ij`，零有效面积格为 NA；边缘低有效面积格可只在显示层隐藏，完整数值仍写入 `density_heatmap_grid.csv`。热图网格总面积和总 count 会与主结果核对，色标、分格和截断状态写入摘要。径向与角向输出包括：`radial_density.csv/png`、`angular_density.csv/png` 和 `regional_density.csv` 分别给出径向、方位角以及 center/middle/edge 的 count、有效面积、密度和泊松区间。默认径向为 `equal_area` 分箱；方位角参考只是图像正 x 轴，未统一晶圆方向时不能解释为晶向。可作为论文 Methods 初稿的操作规则见 [测量协议](docs/measurement_protocol.md)。

## 输出文件

每次分析使用独立目录，例如 `results/sample_001/`：

| 文件 | 含义 |
|---|---|
| `summary.json` / `summary.csv` | 图像、晶圆几何、标定、各类面积、候选数、n、rho、不确定度、95% 区间、版本、耗时、参数和警告 |
| `defects_all.csv` | 全部编号候选，含形态参数、自动决定及拒绝原因 |
| `defects_accepted.csv` | 当前规则接受的候选 |
| `defects_rejected.csv` | 当前规则拒绝的候选；不能删除，因为需要评估漏检 |
| `candidate_crops/` | 每个候选的原始位深 `.tif` 局部裁剪及 `.png` 预览，供追溯/复核；候选特别多时会占用可观磁盘空间 |
| `overlay_accepted.png` | 接受目标的绿色圆圈和编号 |
| `overlay_all_candidates.png` | 接受目标与被拒绝候选（不同符号） |
| `overlay_xrt_red_boxes.png` | 自动接受 XRT 点状候选的红色边界框和物理标尺；没有独立 DIC/KOH 数据时绝不画黄色验证圈 |
| `xrt_detection_detail_montage.png` | 最高 6 个全分辨率代表性局部视场，按真实候选边界框绘制红框并带物理标尺；明确标注独立参考未提供 |
| `paper_detection_field.png` | 单个论文式全分辨率视场；红框为自动候选，黄色圈只来自已经哈希核验和配准的独立参考 |
| `paper_aligned_result_figure.png` | 将上述论文语义视场与整片实际有效面积归一化密度图组合为一张展示成果；不改变计算结果 |
| `independent_reference_points.csv` | 原样保留的独立参考登记表，包含 possible/uncertain 与来源哈希 |
| `independent_reference_matches.csv` | 自动候选与已登记参考点的物理距离匹配审计；参考覆盖不完整时不报告 precision/F1 |
| `defect_comparison_details.png` | 原始局部与自动判定复核；使用同一全局科研灰度窗口，明确不是 DIC/KOH 独立验证，不替代完整候选 CSV 或原始裁剪 |
| `wafer_mask.png` | 分割轮廓优先的完整晶圆区域（拟合圆作为物理标定） |
| `valid_analysis_mask.png` | 面积计算真正使用的最终有效区 |
| `preprocessed_preview.png` | 背景校正/暗响应预览 |
| `candidate_mask.png` | 筛选前候选二值图 |
| `analysis_config.yaml` | 合并 CLI 覆盖值后的本次实际参数 |
| `resolved_physical_parameters.yaml` | 物理单位输入、`um_per_pixel`、转换后的像素参数及物理/旧像素参数来源 |
| `run.log` | 加载方式、警告、失败原因、时间和大图限制 |
| `report.html` | 可在浏览器打开的汇总报告与复核图链接 |
| `*_histogram.png`、`*_distribution.png` | 可选尺寸、径向、角度和散点统计图 |
| `density_heatmap.png` | 整片晶圆点状目标密度热图；每格按 `valid_analysis_mask` 实际有效面积归一化，单位 cm^-2；显示色标可按配置截断但不改定量表 |
| `density_heatmap_grid.csv` | 二维网格的 x/y 边界、有效像素数、有效面积比例、count、密度及逐格 Poisson 95% 区间 |

叠加图在超大图上按 `io.max_overlay_size` 缩小显示，但坐标从原图正确映射，CSV 中仍保留原图全局坐标。候选裁剪则从原图对应 tile 读取，不应拿预览图冒充原始分辨率。

如需像论文图 6 那样并排比较两片晶圆，必须先在两次配置中设置相同的 `spatial.heatmap_vmin_cm2` 与 `spatial.heatmap_vmax_cm2`，再运行：

```bash
python scripts/build_paper_comparison.py \
  results/wafer_a results/wafer_b \
  --output results/paper_comparison.png
```

脚本会拒绝色标不一致的两张热图，避免相同颜色代表不同密度。它沿用“点状目标密度”措辞，不会把论文中的 294、647 或 0–1500 cm^-2 强加给当前晶圆。

## 人工复核

自动分类不是最终物理判定。用 matplotlib 逐个查看候选：

```bash
python scripts/review_results.py \
  /path/to/original_image.tif \
  results/sample_001/defects_all.csv \
  --summary results/sample_001/summary.json \
  --output results/sample_001/reviewed_defects.csv
```

按键：

- `A`：人工接受并前进；
- `R`：人工拒绝并前进；
- `U`：恢复该目标的自动判定；
- 左/右方向键或空格：浏览；
- `S`：保存；
- `Q` 或 Esc：保存并退出。

脚本不覆盖原 `defects_all.csv`，而是写 `reviewed_defects.csv` 和 `reviewed_summary.json`。它从原摘要读取同一 `S_valid`，重新计算接受数、`rho`、`sqrt(n)/S` 和 95% Poisson 区间；若摘要缺少实际有效面积，它会报错，而不会悄悄使用 78.5398 cm²。

## 合成数据和自动测试

模拟生成器固定随机种子，并产生：

- `clean`：均匀背景、高对比、互不粘连的点；
- `noisy`：亮度梯度、随机噪声、扫描条纹和外部黑点；
- `difficult`：另外包含粘连点、长线、大黑斑、文字和标尺伪影。

每张图旁有 `synthetic_<kind>_ground_truth.csv`。其中 `is_true_defect`、`should_count`、`artifact_type` 和每个真实点的中心/半径均由生成器独立保存；检测流程绝不读取真值来生成结果。相同 seed 产生相同像素与标签，便于检查可重复性。运行测试：

```bash
pytest -q
```

测试覆盖毫米/厘米和像素面积换算、100 mm 理论面积、`n=0`、有效掩膜、坐标变换、晶圆检测、点计数、外部/边缘排除、长线不过度拆分、CSV/JSON 字段以及固定种子的重复性。`noisy` 和 `difficult` 应根据独立真值按中心匹配，报告 precision、recall 和 F1；不能读取真值来“检测”。

## 超大图与分块行为

程序先在最长边不超过 `preview_max_size` 的预览图上检测晶圆，再把几何映射回原图。候选检测可按默认 `2048 × 2048` tile 运行，相邻 tile 有 128 px halo；只保留唯一核心区的目标，并通过中心距离/边界框重叠去重，所有坐标最终转换成原图坐标。

不同格式的实际内存行为并不相同：

- 未压缩、可内存映射的 TIFF/BigTIFF：tifffile 可按需访问，适合真正 tile 读取；
- 内置有界 TIFF 后端：可按偏移读取未压缩行/条带，并只解码与目标区域相交且低于安全上限的条带或 tile；网页模式强制使用该后端并禁用整图回退；
- pyvips 可用且格式支持随机访问：使用 pyvips 区域读取；
- 某些压缩 TIFF：原生默认配置仍可能在没有其他后端时完整解码；网页配置不会这样做，压缩单条带或缺少解码器时会明确拒绝；
- PNG/JPG/BMP：OpenCV/Pillow 通常需要解码整张图，虽然之后的检测可以 tile 化，但这不等于文件本身被真正随机分块读取。

程序会在 `summary.json`/`run.log` 中明确写出 loader、`lazy`/`random_access`、`source_region_read_bounded`、`decoded_full_source_resident` 状态和限制。浏览器报告还记录输入传输方式、源/分析分辨率、`analysis_downsample_factor=1` 和是否发生科研降采样。分块能降低检测中间数组峰值，但当前传统形态学背景核在 tile 接缝、极大伪影跨 tile、以及全局坏区标注方面仍需谨慎验证。tile overlap 应至少覆盖最大目标半径、形态学核影响范围和分水岭上下文；修改 `background_kernel_px` 后也要相应检查 overlap。

## 统计不确定度怎么读

报告的 `rho ± sigma_rho` 只使用 Poisson 计数波动。`n=0` 时点估计和标准不确定度均为 0，但 95% Poisson 上限仍为有限正值，因此“没有检测到”不等于真实密度严格为零。

> 该统计不确定度只反映有限计数造成的随机误差，不包含图像分割、漏检、误检以及物理判定错误造成的系统误差。

分割阈值、晶圆标定、漏检/误检、图像伪影和“黑点是否为位错”的判断都属于系统不确定度，需要标注验证、重复成像或独立实验评估。

## 当前限制与真实样品接入

- 这是基于灰度、形态和局部对比度的传统算法；在目标对比度接近噪声、背景纹理尺度与目标相同或目标形态发生域偏移时会失效。
- 自动无效区目前依赖明显图像规则和配置；复杂遮挡、手写标记或未知设备伪影可能需要人工掩膜。
- 平边/缺口过大、圆周出框或输入只是内部裁片时，自动圆拟合可能不可靠，必须人工标定。
- 分水岭只能分开有可辨距离峰的粘连点；完全重叠或严重模糊的点没有足够图像信息。
- 默认阈值尚未经过导师标注的真实 SiC 数据校准，不能直接作为论文中的普适判据或性能结论。
- 程序不会声称“每个识别点一定是一条真实位错”。报告始终使用“当前判定标准下的点状缺陷/点状目标”措辞。

接入真实图像时，最好同时提供：原始未压缩或无损图、图像是否为整片晶圆、像素/物理标尺或设备几何、晶圆圆周/平边/缺口信息、需要排除的边缘宽度、已知遮挡/文字/标尺区域、典型点直径范围（px 或 µm）、材料专家逐点接受/拒绝标注，以及至少一批不参与调参的独立验证图。若多张图只覆盖局部视场，还需要每张视场在晶圆上的位置与有效视场掩膜，不能把局部面积误当作整片面积。
