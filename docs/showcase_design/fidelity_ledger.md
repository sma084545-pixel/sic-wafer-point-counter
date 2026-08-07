# Concept-to-implementation fidelity ledger

The concept PNGs are interface references, not scientific evidence. The two
public `implementation-result-*.png` files were captured from the running local
Flask app on the fixed-seed clean synthetic analysis. A previous overview
capture was intentionally excluded from the public snapshot because its recent
run table contained a real sample filename.

| Dimension | Concept intent | Implemented result | Verdict |
|---|---|---|---|
| Information hierarchy | Scientific boundary first, then recent result and data chain | Same order on the overview; result pages lead with n, S, rho, counting uncertainty, and validation state | Matched |
| First viewport | Stable navigation rail with a dense but calm work area | 232 px rail at 1440 and 1024 px; primary result and warning remain above the fold | Matched |
| Typography | System sans, compact headings, tabular scientific values | System Chinese fallbacks, balanced headings, tabular number columns, scientific notation | Matched |
| Colour | Cool grey surfaces, restrained SiC teal, amber warnings | Token values in `app.css` reproduce the concept hierarchy without gradients | Matched |
| Spacing | Tight instrument rhythm rather than marketing whitespace | 4 px base rhythm with 8–32 px gaps and dense metadata rows | Matched |
| Table density | Real tabular structure for run and candidate data | Desktop candidate table shows 50 server-paged rows; mobile alone switches to records | Matched |
| Image ratio | XRT output remains the visual anchor | Main overlay uses a dark bounded viewer and `object-fit: contain`; no generated image is presented as data | Matched |
| State styling | Words plus colour for completed, failed, warning, and unvalidated | Green completion, red rejected-output state, amber unvalidated state, all with explicit text | Matched |
| Mobile layout | Compact header, 2-column metrics, stacked review flow | Verified at 390×844 with no core horizontal overflow and an Escape-close navigation drawer | Matched |
| Motion | Minimal, functional progress only | One indeterminate transform/position animation; reduced-motion disables it | Matched |

Intentional differences:

- Concept-only account and settings controls were omitted because this is a
  single-user loopback application.
- Real result density and metadata are more compact than the concept so the
  10048×10171 XRT preview remains visible without hiding required fields.
- Scientific chart colours are produced by the existing analysis outputs and
  were not recoloured in the browser, preserving report reproducibility.

## Static GitHub Pages adaptation

The public static page at `docs/index.html` reuses the accepted instrument
system as a read-only project introduction. It never simulates upload or Python
analysis in the browser. The latest render was checked at 1440×900 and 390×844
against `concept-overview-desktop.png`, `concept-result-desktop.png`, and the
two public clean-synthetic implementation captures.

| Comparison point | Reference evidence | Static render evidence | Outcome |
|---|---|---|---|
| Scientific hierarchy | Boundary and rho definition lead the application concepts | Hero leads to the boundary panel before feature claims | Matched |
| Palette and type | White/cool-grey surfaces, SiC teal, compact system sans | Exact shared tokens and system Chinese fallbacks | Matched |
| Data integrity | Result concept separates n, S, rho, interval, and validation | Fixed-seed clean values are shown separately with an explicit real-validation warning | Matched |
| Image treatment | Wafer result remains the visual anchor | Clean-synthetic desktop result is the only above-fold data image | Matched |
| Responsive structure | Mobile concept stacks metrics and imagery without tiny tables | 390×844 has zero document overflow and native-width media | Matched |
| Accessibility | Existing design requires landmarks, skip link, focus and alt text | One H1, semantic landmarks, visible focus, descriptive figures and reduced motion | Matched |

The above-the-fold copy differs intentionally because this page explains and
distributes the project rather than operating the analyzer. No account,
settings, upload, or fake interactive controls were introduced. The public
snapshot excludes real source images, paths, run outputs, candidate crops and
the earlier overview capture that exposed a real sample filename.
