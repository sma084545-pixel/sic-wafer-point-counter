# Local scientific showcase platform

## Positioning

This is a local “research showcase + analysis submission + result review”
surface for the existing SiC XRT point-like-target pipeline. It is not a
marketing site, a replacement detector, or a public data service. Every new
analysis still calls `sic_wafer_counter.pipeline.analyze_image`.

The platform reports targets that satisfy the current image rules. It does not
claim that every target is a physical dislocation. Real precision, recall, F1,
classification uncertainty, and a dislocation interpretation remain
unavailable until expert-labelled real SiC data are supplied.

The GitHub Pages companion at `docs/analyze.html` adds a bounded browser-only
path. It downloads a pinned Pyodide runtime, verifies the packaged project
wheel by SHA-256, and calls the same `analyze_image` function in a Web Worker.
The original `File` is mounted read-only through WORKERFS instead of being
copied into main-thread and MEMFS buffers. Files may be selected up to 100 MiB,
but release is format-tiered: raster formats retain the 24 MiB / 6 MP / 6000 px
full-array limits, while compatible scalar 2-D TIFF/BigTIFF may use the bounded
random-access backend up to 100 MiB / 120 MP / 16000 px and are analysed at
source resolution with 1024 px overlapping tiles. Unsupported axes, codec or
segment layouts, absent bounded access, oversized result archives, and memory
failures stop without displaying density; they never trigger hidden downsampling
or full-image TIFF fallback.

## Information architecture

1. **Overview** — scientific boundary, `rho = n / S`, latest persistent result,
   scientific/display data flow, and recent runs.
2. **New analysis** — local image selection, 100 mm diameter, edge exclusion,
   threshold, watershed, optional manual circle, and fixed-seed demos.
3. **Run history** — restart-persistent completed, failed, synthetic, and real
   exploratory records; malformed summaries are isolated.
4. **Result detail** — n, S, rho, counting uncertainty, Garwood interval,
   grayscale precision, geometry, actual mask areas, warnings, artifacts,
   candidates, and area-normalized spatial plots.
5. **Methods & validation** — scientific channel, physical parameter
   resolution, mask area, validation split rules, and uncertainty boundaries.

## Data sources and trust boundaries

- Quantitative values come from one run's `summary.json`.
- Candidate rows come from that run's `defects_all.csv`.
- Images and downloadable audit files must already exist inside that run
  directory. A browser request never opens the `input_path` recorded in a
  summary.
- Synthetic source images are generated with a fixed seed, then passed through
  the real pipeline. Ground truth is used by tests and validation only.
- Missing old-summary fields render as “未提供”; JavaScript does not turn
  `null` into a scientific zero.

## API

| Method and path | Purpose |
|---|---|
| `POST /api/jobs` | Validate one upload and enqueue a real pipeline run |
| `GET /api/jobs/<job_id>` | Return transient `queued/running/completed/failed` state |
| `GET /api/jobs/<job_id>/files/<path>` | Compatibility artifact route for an in-memory job |
| `POST /api/demo/<clean|noisy|difficult>` | Enqueue an existing fixed-seed synthetic image |
| `GET /api/runs` | Discover valid direct result directories after restart |
| `GET /api/runs/<run_id>` | Return a sanitized summary and available artifact URLs |
| `GET /api/runs/<run_id>/files/<path>` | Return one allowlisted in-run file |
| `GET /api/runs/<run_id>/defects` | Stream a filtered candidate page |

Candidate query parameters are `status`, `reason`, `defect_id`, `page`, and
`page_size`. Page size is limited to 1–200. The endpoint scans CSV rows without
building a 100,000-row JSON object and returns only the requested page, total,
page count, and rejection-reason counts.

## Safety and privacy model

- The CLI server rejects any host other than `127.0.0.1`, `localhost`, or
  `::1`; the launcher always uses `127.0.0.1`.
- Upload names use an image suffix allowlist and `secure_filename`; size is
  limited before Flask accepts the request.
- A run ID is a restricted direct-child identifier. File paths reject absolute
  paths, empty/dot components, `..`, disallowed suffixes, and any symlink.
- Summary path values are reduced to basenames before entering JSON responses.
- Responses set CSP, `nosniff`, no-referrer, same-origin resource policy, and
  frame denial; API responses use `Cache-Control: no-store`.
- No CDN, remote font, analytics, cloud image service, arbitrary YAML/Python,
  shell endpoint, deletion control, or public bind is present.
- `results/`, `web_uploads/`, `.venv/`, generated samples, and browser audit
  output are Git-ignored. Tracked implementation screenshots contain only
  synthetic analysis data.

## Performance strategy

- `max_workers=1` prevents overlapping large analyses.
- The overview requests a compact run index and one existing PNG, never the
  original TIFF.
- The result view has one active main image. Other scientific figures are lazy
  decoded, candidate crops are lazy loaded, and dimensions are explicit.
- Candidate CSV access is constant-memory with respect to result size; the UI
  holds one page and renders at most 200 rows.
- Missing crops do not trigger TIFF reads or on-demand image processing.
- The scientific pipeline retains its preview/tile/memory-map/optional-pyvips
  behaviour. Browser TIFF runs force `prefer_bounded_tiff_regions`, disable
  memory-map and full-decode fallbacks, and require audit metadata proving
  `source_region_read_bounded=true` and
  `decoded_full_source_resident=false` before a result is exposed.
- Large candidate CSVs remain complete in the ZIP but are not also transferred
  as individual browser buffers; small report and preview artifacts remain
  directly viewable.

## macOS runtime

`scripts/run_local_web_workbench.sh` resolves the project-local `.venv` first
and imports the current checkout through `PYTHONPATH=src`. The per-user
installer creates `org.sic-wafer-counter.local-workbench` with `RunAtLoad` and
`KeepAlive`, so the service does not depend on an open terminal and is restored
after login. Both the development launcher and LaunchAgent remain loopback
only.

## Browser and accessibility verification

The implemented UI was exercised in real Chrome/Chromium at 1440×900,
1024×768, and 390×844. Checks covered the overview, form validation, real
result, low-confidence failure, 100k-scale pagination, image gallery, mobile
navigation, skip link, visible focus, live task status, one visible H1, image
alt text, labelled controls, no external assets, and no core horizontal
overflow. Candidate tables remain tables on desktop and become compact records
only on narrow screens.

The implementation uses ordinary semantic HTML, CSS, and ES modules. Current
Safari and Firefox are expected to support the core flow, but this release did
not perform equivalent visual regression captures in those browsers.

### Browser large-TIFF release validation

Version 0.2.1 was exercised end to end in Chrome 151 against the locally served
GitHub Pages files. The deterministic fixture was a 93.5 MiB, 7000×7000,
uint16, scalar `YX`, uncompressed one-strip TIFF containing 96 synthetic dark
points. The browser completed 45 source-resolution tiles in 809.585 seconds on
the validation machine and reported the same count, valid area, and density as
the native run: `n=96`, `S=78.5392950052 cm²`, and
`rho=1.2223180765 cm^-2`. Its summary recorded
`WORKERFS_read_only_File`,
`tifffile.bounded-regions(direct-uncompressed-strips)`,
`source_region_read_bounded=true`,
`decoded_full_source_resident=false`, float32 analysis,
`analysis_downsample_factor=1`, and no scientific downsampling. The 6,372,496
byte ZIP contained 29 files, passed CRC testing, and included all three defect
tables, mask, red-box/detail views, area-normalized heatmap/grid, HTML report,
and summary.

This validates that exact layout and execution path; it is not a claim that
every TIFF near 100 MiB is supported, nor evidence of accuracy on real SiC.
Unsupported codecs, oversized compressed segments, multidimensional series,
and environments without WORKERFS/bounded access remain fail-closed and should
use the native workbench.

## Demonstration

Generate source images if needed:

```bash
.venv/bin/python scripts/generate_synthetic_wafer.py --all --output-dir sample_data/generated
```

Start the platform:

```bash
./scripts/run_local_web_workbench.sh --open
```

Choose **新建分析 → Clean**. A successful run should accept 96 targets and
display the actual valid-mask area and `rho`; these are synthetic-pipeline
results, not real SiC accuracy evidence.

## Known limitations

- Candidate filtering scans the CSV for each request. Memory remains bounded,
  but repeated deep-page requests on extremely large tables may be I/O-bound.
- There is no cancellation endpoint; a submitted analysis finishes or fails in
  the single local worker.
- The development Flask server is intentionally local and is not a production
  WSGI/public deployment.
- Uploads are retained for audit and are not automatically deleted.
- The interface does not edit `accepted`; use `scripts/review_results.py` for
  reviewed CSV and density recalculation.
- Real XRT thresholds, invalid masks, and physical interpretation still require
  material-expert calibration and locked-wafer validation.

Design concepts, tokens, implementation captures, and the fidelity comparison
are in `docs/showcase_design/`.
