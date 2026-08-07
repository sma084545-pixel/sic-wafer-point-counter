# Academic measurement baseline

## Quantity currently reported

The program reports the density of black point-like image targets accepted by
the current image rules:

`rho = n / S`, in `cm^-2`.

`n` is the number of automatically accepted point-like candidates.  `S` is the
area of the final `valid_analysis_mask`, calculated as its valid pixel count
times the calibrated pixel area.  It is not replaced by the ideal 100 mm wafer
area.

## Current verification scope

Verification is currently limited to deterministic synthetic wafer images and
algorithmic unit tests.  It verifies geometric calibration, mask-area
accounting, outside/edge exclusion, line rejection, reproducibility, required
output files, and synthetic point matching.  It does not establish accuracy on
real SiC XRT data.

## Current uncertainty scope

The reported standard uncertainty is `sqrt(n) / S`; the reported interval is a
Garwood Poisson count interval scaled by `S`.  These describe finite-count
random variation only.  They do not quantify segmentation, classification,
parameter-selection, pixel-scale, invalid-mask, or physical-identification
uncertainty.

## Claims not established

An accepted black image target is not automatically a physical dislocation.
No real-SiC precision, recall, F1, classification uncertainty, or material
mechanism claim is established without independent expert annotations or
experiments.

## Baseline test result

- Software version: `0.1.0`
- Pre-existing Git commit: none; this directory was not a Git repository when
  the baseline was recorded.
- Full test suite: `13 passed`.
- Clean synthetic image: 96 accepted targets; precision/recall/F1 =
  1.000/1.000/1.000.
- Noisy synthetic image: precision/recall/F1 = 1.000/1.000/1.000.
- Difficult synthetic image: precision/recall/F1 = 1.000/0.988/0.994.
- Clean analysis output was created at `results/academic_baseline_clean` and
  includes the required summary, defect tables, valid-area mask, accepted
  overlay, and HTML report.
