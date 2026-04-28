# Export notes — `Thesis export/`

Decisions, caveats, and conventions captured during the export of `revised_manuscript_2.tex` so the next agent (or the author) can transform `text.md` into a thesis chapter without re-deriving context.

---

## 1. Source provenance

- Source manuscript: `g:\My Drive\PAPERS\DATACENTERS IMPERIAL\DCFlex\MANUSCRIPT\revised_manuscript_2.tex`
- Latest commit touching this file: `8727069f4b6ee71234137791012f6158fde7933a` ("hardcoded bibliography due to buggy EM. Finally passed check")
- Source line count: 2,006
- Generated `text.md`: 1,045 lines
- Generated `references.bib`: 54 entries (738 lines)
- Figures copied: 29 PNGs (~18.6 MB)
- The original `MANUSCRIPT/` folder was not modified. This export folder is additive.

The conversion script lives at `_build/tex_to_md.py` and is reproducible: `python "Thesis export/_build/tex_to_md.py"` regenerates `text.md` and `references.bib` deterministically from the read-only inputs.

---

## 2. Conventions to preserve

These project-specific conventions appear in figure file names, captions, and prose. The thesis-writing agent must keep them consistent if it reuses the figures or refers to the analysis outputs.

| Concept | Convention | Where it surfaces |
|---|---|---|
| Scenario names | `BASELINE`, `DAM`, `mFRR` | analysis pipeline, figure file names, joblib exports |
| **Display name vs internal name** | The `DAM` scenario is presented to readers as **INTRADAY**. | Figures named `..._DAM.png` are referenced as the Intraday scenario in prose. |
| ZETA suffix | `ZETA_0`, `ZETA_010`, `ZETA_020`, `ZETA_030`, `ZETA_050`, `ZETA_070` (zeros padded) | input CSV filenames and export filenames |
| Sample week | 1970-02-01 → 1970-02-07 (Unix epoch placeholder; the trace was anonymized 2019–2020) | the sample-week analysis throughout `text.md` |
| Colorblind palette | BASELINE = `#0571B0` (blue); DAM = `#FC9272` (red); mFRR = `#A6D854` (green) | comparison figures |
| Notable points | P1, P2, P3 (top-3 deferrable energy); Q1, Q2, Q3 (75th percentile); R1, R2, R3 (median) | scenario subsections |
| Latency parameter | $\hat{\lambda}$ (10/3/30 min for Baseline / mFRR / Intraday) | tables `tab:scenario_parameters`, `tab:parameters_*` |
| GPU power model | Class-based, linear in GPU utilization (no telemetry) | Methodology Section 3 |

---

## 3. Bibliography

- **Conversion path:** the manuscript shipped with an inline `\begin{thebibliography} … \end{thebibliography}` (lines 1575–2002 of the .tex) — a hardcoded fallback added in commit `8727069` to pass an editorial-manager check. The structured BibTeX file `MANUSCRIPT/bibliography_2.bib` was orphaned.
- **What the export does:** for each of the 54 cited keys, `references.bib` reuses the structured `bibliography_2.bib` entry verbatim (with the cited key as the entry ID, even when the .bib used an alias such as `Zhang2024_SusComp` for `ZHANG2024DR`). When a cited key has no .bib match, the script synthesizes a `@misc` entry from the free-form `\bibitem` text and prepends `% source: \bibitem (free-form parsed -- review fields)`. **Run-result for this paper:** zero synthesized entries — all 54 keys had structured matches.
- **Aliases applied:** `Zhang2021_DATE`↔`HPC`, `ZHANG2024DR`↔`Zhang2024`/`Zhang2024_SusComp`, `Zhang2025`↔`Zhang2023`, `Zheng2020`↔`zheng2020mitigating`, `Takci2025`↔`takci2025data`.
- **Citations in `text.md`** are emitted as `[@key]` and `[@key1; @key2]` (Pandoc citeproc syntax). A thesis build using `pandoc --citeproc` with `references.bib` will render them.

The 14 entries of `bibliography_2.bib` that were never cited (7 unused + 7 alias duplicates) are **not** copied into `references.bib`.

---

## 4. Figures

- All 29 figures in `figures/` are byte-for-byte copies of the originals in `MANUSCRIPT/`. No renames, no recompression.
- Largest file: `methodology.drawionew.png` (4.2 MB). For thesis printing, consider re-exporting from the original draw.io source as PDF; the thesis template may benefit from vector rather than raster.
- **Subfigure pairs to keep together:**
  - `A.png` + `B.png` — CDF + scatter (parent label `fig:cdf_duration`).
  - `a_posteriori_interval1.png` + `a_posteriori_interval2.png` — stacked time series, full vs flexibility-range (parent label `fig:interval_comparison`).
- **Figures dropped from export** (present in `MANUSCRIPT/` but never referenced in the .tex):
  - `FlexibilityWindow.png`, `LAD_flex_default_3x3_plot_ALT_LARGE.png`, `LAD_flex_default_P1_explanation.png`, `LAD_flex_default_points_LARGE.png`.
- **Commented-out reference dropped:** `latency definition.png` was inside a `%`-prefixed block at line 432 of the .tex; the active equivalent is `latencydiagram.png` (line 439, label `fig:latency`).

---

## 5. Anchors and cross-references

- LaTeX `\label{lab}` is emitted in `text.md` as `<a id="lab"></a>` immediately above the host element (figure / table / equation / case clause).
- LaTeX `\ref{lab}` becomes `[lab](#lab)`; `\eqref{lab}` becomes `([lab](#lab))`. The visible link text is the raw label name — the thesis-writing agent may want to render these as "Section 3.1" / "Eq. (4)" once the chapter numbering is fixed.
- A few labels contain spaces (e.g. `\label{case study}` → `<a id="case study"></a>`). HTML accepts these but Markdown anchor links may URL-encode them. Acceptable as-is; rename if a renderer complains.

---

## 6. Equations

- Inline math kept verbatim in `$...$`.
- Display equations rendered as `$$ ... $$` with the original `\label{}` exposed as a preceding HTML anchor.
- The four-clause `\begin{numcases}{Deferrable_i \iff} … \end{numcases}` block (Section 3, deferrability conditions) was rendered as a `cases` environment inside `$$…$$`. Each clause's `\label{case:start}` etc. is preserved as a separate anchor. The right-hand `\text{…}` annotations ("Start after notification", "End before buffer", …) are kept inside the math.
- Some inline math uses `\( … \)`; this is valid in MathJax/KaTeX. If the thesis pipeline rejects it, normalize to `$ … $`.

---

## 7. Tables

- **Simple 2-column** definition tables (the seven nomenclature blocks, the column-description table) are converted to Markdown pipe tables.
- **Simple multi-column** numeric tables (e.g. `tab:specs`, scenario-parameter tables) are also Markdown tables.
- **Complex tables** with `\multicolumn`, `\cmidrule`, `\diagbox`, or nested `tabular*` are emitted as fenced `\`\`\`latex \`\`\`` blocks under their caption + anchor. They appear in `text.md` for these tables: `tab:energy_flexibility_baseline`, `tab:energy_flexibility_intraday`, `tab:energy_flexibility_mfrr`, `tab:latency_distribution`, `tab:deferrable_energy_latency`, `tab:scenario_summary`, `tab:opt_delta_and_revenue_transposed`, `tab:comparison_optimal_fixed`. The thesis-writing agent should re-render these manually for the target template.

---

## 8. Dropped content

- `\begin{thebibliography}` block — replaced by `references.bib`.
- The bibliography URL/href shim at TeX lines 1576–1580 — irrelevant in Markdown.
- `\arraystretch` / `\setlength` table-spacing tweaks — no Markdown analog.
- `\renewcommand{\labelitemi}{$\bullet$}` — Markdown `-` already produces a bullet.
- `\centering`, `\noindent`, `\vspace`, `\hspace`, `\bigskip`, etc. — purely typographical, dropped.
- The Elsevier author-email macros `\ead{…}` were stripped (emails aren't typically displayed in thesis chapters).
- Commented-out figure reference at TeX line 432.
- One stray bare `\` at TeX line 368 (pre-existing typo in the source).

---

## 9. Acknowledgements and declarations

`text.md` keeps the three `\section*{…}` declarations that closed the paper (Declaration of competing interest, Availability of data and materials, Acknowledgements, Declaration of generative AI). For a thesis chapter most of these need to move:

- Acknowledgements → thesis front matter.
- Declaration of competing interest → drop (journal-specific).
- Availability of data → optionally retain or move to a thesis-wide section.
- Declaration of generative AI → drop (journal-specific).

---

## 10. Unrecognized constructs

`_build/unrecognized.log` records LaTeX commands the conversion script could not map. **For this paper the log is empty.** That is consistent with the inspection report: the manuscript contains no user-defined macros and only a small set of standard packages.

---

## 11. Reproducing the export

```bash
# from project root
python "Thesis export/_build/tex_to_md.py"
```

The script reads `MANUSCRIPT/revised_manuscript_2.tex` and `MANUSCRIPT/bibliography_2.bib` and writes only into `Thesis export/text.md`, `Thesis export/references.bib`, and `Thesis export/_build/unrecognized.log`. Re-running is idempotent for a given input; figure copies in `Thesis export/figures/` are not re-derived by the script and were placed once at export time.
