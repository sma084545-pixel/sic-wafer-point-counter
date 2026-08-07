# Showcase visual system

This visual system translates the three concept images in this folder into a
local, code-native Flask/Jinja interface. The concept images are design
references only; scientific values shown by the application always come from
real pipeline output.

## Product character

- Scientific instrument, not marketing site.
- Dense enough for research review, but calm and legible.
- Every state distinguishes measured data, synthetic demonstrations, missing
  validation, warnings, and failures with both words and colour.
- No gradients, glass effects, decorative illustrations, external fonts, or
  external CDN assets.

## Tokens

| Role | Value |
|---|---|
| Page | `#f5f7f7` |
| Surface | `#ffffff` |
| Surface muted | `#eef3f2` |
| Text | `#172329` |
| Muted text | `#596970` |
| Border | `#d6dfdd` |
| Teal | `#087f83` |
| Teal dark | `#056166` |
| Teal pale | `#e5f3f2` |
| Amber | `#ad6800` |
| Amber pale | `#fff7e6` |
| Error | `#b42318` |
| Error pale | `#fff0ee` |

Spacing follows a 4 px base with primary gaps of 8, 12, 16, 24, and 32 px.
Corners are restrained: 6 px controls, 8 px cards, 10 px large panels. Borders
are 1 px; shadows are reserved for the mobile navigation drawer and alerts.

## Typography

Use the operating-system sans-serif stack with Chinese system fallbacks.
Headings are compact and semibold. Numerical values use tabular numerals.
Paragraph width is capped near 76 characters. The interface remains usable at
200% text zoom.

## Layout

- Desktop: 232 px navigation rail plus one fluid work area.
- Main result view: metrics row, image/metadata split, then candidate browser.
- Mobile: app bar, two-column metrics, responsive image, stacked candidate
  records, fixed four-item bottom navigation.
- Wide data tables live inside labelled horizontal scroll regions; core actions
  never require horizontal scrolling.

## Interaction and accessibility

- Native buttons, inputs, fieldsets, tables, and dialog elements are preferred.
- A skip link leads to the main region.
- Focus uses a visible 3 px teal outline with 2 px offset.
- Touch targets are at least 44 px on mobile.
- One polite live region reports normal updates; errors use a visible alert.
- Indeterminate analysis progress never implies a fabricated percentage.
- Reduced-motion preference disables decorative and progress animations.

## Reference concepts

- `concept-overview-desktop.png`: information hierarchy and recent-run table.
- `concept-result-desktop.png`: metrics, image review, metadata, and pagination.
- `concept-result-mobile.png`: responsive ordering and stacked candidates.

The implementation intentionally omits concept-only account/settings controls
because the platform is single-user and local-only.
