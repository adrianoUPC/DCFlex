# Paper → Thesis-Chapter Mapping

This file is a **scaffold** for the next stage. The left column lists every section the paper actually contains (extracted from the compiled `revised_manuscript_2.tex`). The middle column is empty — fill it with the target thesis chapter and section. The notes column flags overlaps with other thesis chapters, decisions to make, or content to drop.

When the thesis structure is decided, the next agent (or the author) edits the right two columns and uses `text.md` as the prose source.

---

## Paper title

**Data Center Workload Flexibility for Power System Demand Response: Evidence from Alibaba Traces**

Provenance: `MANUSCRIPT/revised_manuscript_2.tex`, latest commit `8727069 hardcoded bibliography due to buggy EM. Finally passed check`.

---

## Mapping table

| Paper section (as in `text.md`) | Thesis chapter target | Notes |
|---|---|---|
| Title, Authors, Affiliations | _to fill_ | Move authorship to thesis front matter; affiliations probably drop. |
| Abstract | _to fill_ | Likely rewritten as the thesis chapter abstract / opening. |
| Keywords | _to fill_ | Drop or fold into thesis index keywords. |
| Nomenclature → Abbreviations | _to fill_ | Merge into a global thesis Nomenclature appendix. |
| Nomenclature → Indices and Sets | _to fill_ | Merge into global Nomenclature. |
| Nomenclature → Task Variables | _to fill_ | Merge into global Nomenclature. |
| Nomenclature → Node-Level Parameters | _to fill_ | Merge into global Nomenclature. |
| Nomenclature → Data Center-Level Variables | _to fill_ | Merge into global Nomenclature. |
| Nomenclature → Flexibility Strategy | _to fill_ | Merge into global Nomenclature. |
| Nomenclature → Flexibility Optimization | _to fill_ | Merge into global Nomenclature. |
| 1. Introduction → Background and motivation | _to fill_ | High overlap with thesis Ch. 1 (motivation). Trim duplicated ground. |
| 1. Introduction → Data centers as flexibility providers | _to fill_ | High overlap with thesis Ch. 1 / Ch. 2. |
| 1. Introduction → Literature Review | _to fill_ | Strong candidate for **thesis Ch. 2 (state of the art)**. May merge with literature reviews from other papers. |
| 1. Introduction → Research Gap and Contribution | _to fill_ | Trim — thesis-level contributions stated once, in the thesis intro. |
| 1. Introduction → Paper Organization | _to fill_ | **DROP** — replaced by chapter intro. |
| 2. Problem Description → Operative conditions | _to fill_ | Body of the chapter. |
| 2. Problem Description → Flexibility provision | _to fill_ | Body of the chapter. Includes Fig. methodology_diagram.png. |
| 3. Methodology → "A posteriori" flexibility assessment | _to fill_ | Core methodology section. Equations `eq:power_DC1`, `eq:power_DC2`, `eq:energy_group`, `eq:power_group` — keep. Fig. latencydiagram.png. |
| 3. Methodology → LAD-Flex → Aggregator Flexibility Window | _to fill_ | Core. Equations for `t_1`, `t_2`. |
| 3. Methodology → LAD-Flex → Latency threshold | _to fill_ | Core. |
| 3. Methodology → LAD-Flex → Incorporating task duration prediction uncertainty | _to fill_ | Core. Stochastic model `\hat{d}_k = d_k + \epsilon_k`. |
| 3. Methodology → LAD-Flex → Flexibility window parameters: overhang and buffer period | _to fill_ | Core. `numcases` block with deferrability conditions (anchors `case:start`, `case:end`, `case:start_overhang`, `case:end_overhang`). Fig. task_flex_conditions.png. |
| 3. Methodology → Notification period optimization | _to fill_ | Core. Eqs. `smooth_decay`, `eq:max_returns`. |
| 4. Case Study → Case study assumptions | _to fill_ | Body. Table `tab:specs` (cluster config), Fig. cdf_duration (subfig A.png + B.png), Table `tab:task_summary_duration`. |
| 4. Case Study → Flexibility Scenarios and Market Context → Baseline Scenario | _to fill_ | Body. Table `tab:scenario_parameters`. |
| 4. Case Study → Flexibility Scenarios and Market Context → mFRR Scenario | _to fill_ | Body. |
| 4. Case Study → Flexibility Scenarios and Market Context → Intraday Scenario | _to fill_ | Body. |
| 4. Case Study → Flexibility bid requirements | _to fill_ | Body. Fig. flexibility_value_function_color.png. |
| 5. Results → "A posteriori" flexibility assessment | _to fill_ | Body. Fig. interval_comparison (subfig a_posteriori_interval1.png + a_posteriori_interval2.png), Table `tab:latency_distribution`. |
| 5. Results → LAD-Flex strategy results → Baseline Scenario | _to fill_ | Body. 3 figures: `LARGE_DEFAULT_BASELINE_ZETA_0.png`, `power_consumption_3x3_LARGE_DEFAULT_BASELINE.png`, `LAD_flex_P1_explanation_DEFAULT_BASELINE.png`. Table `tab:energy_flexibility_baseline`. |
| 5. Results → LAD-Flex strategy results → Intraday Market Scenario | _to fill_ | Same shape as Baseline; 3 figures + 1 table. **Note conv:** displayed as "INTRADAY" but file/code naming uses `DAM`. |
| 5. Results → LAD-Flex strategy results → mFRR Balancing Reserve Scenario | _to fill_ | Same shape; 3 figures + 1 table. |
| 5. Results → LAD-Flex strategy results → Scenario Comparison | _to fill_ | 5 figures (`time_series_overlay_sample_week`, `combined_comparison_publication_ready`, `notable_points_comparison`, `hourly_yield_comparison`, `rebound_energy_summary`), 1 summary table `tab:scenario_summary`. Eqs. `eq:hourly_yield`, `eq:rebound_ratio`. |
| 5. Results → LAD-Flex strategy results → Sensitivity Analysis | _to fill_ | 2 figures (`sensitivity_grid_fixed_shading_distinct_latencies`, `sensitivity_KDE_latency_filtered_improved`), 1 table `tab:deferrable_energy_latency`. |
| 5. Results → LAD-Flex strategy results → Robustness Analysis Under Duration Prediction Uncertainty | _to fill_ | 3 figures, 1 table `tab:prediction_error_scenarios`. |
| 5. Results → Notification period optimization | _to fill_ | 1 figure (`flexibility_sensitivity_all_points_impr_clean`), 2 tables (`tab:opt_delta_and_revenue_transposed` — complex with `\diagbox`; `tab:comparison_optimal_fixed`). |
| 6. Limitations and future work | _to fill_ | Likely merge into a thesis-wide limitations chapter or a closing chapter section. |
| 7. Conclusions | _to fill_ | Probably trim — thesis has its own concluding chapter. |
| Declaration of competing interest | _to fill_ | **DROP** — belongs to journal submission only. |
| Availability of data and materials | _to fill_ | Optionally re-state in thesis acknowledgements. |
| Acknowledgements | _to fill_ | Move to thesis acknowledgements (front matter). |
| Declaration of generative AI and AI-assisted technologies | _to fill_ | **DROP** — journal-specific. |

---

## Cross-paper linking (placeholder)

If the thesis bundles three papers, list shared concepts here so the thesis-writing agent can detect overlap and consolidate:

- _shared concept_ → also covered in: _other paper_
- _shared concept_ → also covered in: _other paper_

---

## Decisions still open

1. **Nomenclature**: keep per-chapter or fold into a single thesis-wide nomenclature section?
2. **Literature review**: split out as standalone Ch. 2, or distribute across each chapter's introduction?
3. **Conclusions**: keep a per-chapter "Chapter conclusions" or rely on the thesis-level Conclusions?
4. **Limitations**: keep section per-chapter or aggregate at the end of the thesis?
