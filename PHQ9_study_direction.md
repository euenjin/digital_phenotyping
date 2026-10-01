# PHQ-9 Study Direction

## Working Direction

The study direction is now centered on the association between resting heart rate and PHQ-9 depressive symptom severity. This is a cohort-preparation and association-analysis project, not a multimodal incremental machine-learning study.

## Proposed Study Framing

Working title:

Resting heart rate and PHQ-9 depressive symptom severity among Korean adults in KNHANES 2014-2024

Core question:

Is resting heart rate associated with PHQ-9 depressive symptom severity among Korean adults after preparing an appropriate KNHANES analytic cohort?

## Outcome Definitions

Primary outcome variables:

- `PHQ_sum`: PHQ-9 total score, 0-27.
- `PHQ_group`: numeric three-level severity group.

Main outcome:

- PHQ 0-4 / 5-9 / >=10 three-group depressive symptom severity.

Current three-group coding:

- PHQ 0-4: normal/minimal
- PHQ 5-9: mild
- PHQ >=10: moderate or higher depressive symptoms

## Predictor Blocks

The cleaned dataset retains these covariates for later descriptive and regression analysis:

- Demographic/socioeconomic: age, sex, income, education, marital status, employment, survey year.
- Self-report stress: `BP1_reversed`.
- Physiological/clinical: `pulse_rate_60`, systolic BP, diastolic BP, fasting glucose, BMI.
- Sleep: weekday sleep, weekend sleep, weighted average sleep.
- Activity: walking, vigorous, moderate, transport activity minutes.

The primary exposure is `pulse_rate_60`, with `pulse_group` and `pulse_per10` prepared for later interpretation.

## Current Cohort Preparation

`1_data_cleaning.py` now creates:

- `phq_complete`, `PHQ_group`
- `pulse_complete`, `pulse_valid`, `pulse_group`, `pulse_per10`
- `adult_flag`
- disease exclusion indicators for hypertension, stroke, MI/angina, and thyroid disease
- `disease_info_complete`
- `survey_design_complete`
- `primary_cohort`
- `df_primary`
- `cohort_flow`

Current primary cohort output:

- Final primary analytic cohort: 15,617 participants.
- PHQ_group 0: 12,669.
- PHQ_group 1: 2,162.
- PHQ_group 2: 786.
- Valid resting pulse range in the primary cohort: 40-200 bpm.

## Suggested Analysis Order

1. Descriptive table by PHQ group.
2. Group comparison tests for sleep, activity, pulse rate, stress, and sociodemographic factors.
3. Resting heart rate summaries by PHQ group.
4. Later regression analysis for the association between resting heart rate and PHQ group.
5. Careful wording emphasizing association rather than causality.

## Interpretation Guardrails

- KNHANES is cross-sectional, so causal language should be avoided.
- PHQ-9 is a symptom severity score, not a formal clinical diagnosis by itself.
- Disease exclusions should not treat codes 8, 9, or missing as disease-free.
- Survey weights are retained, but final pooled-year weights have not been calculated in the cleaning script.

## Immediate Codebase Implication

The codebase is now aligned with this direction:

- `1_data_cleaning.py` creates `PHQ_sum`, `PHQ_group`, pulse flags, disease exclusions, survey-design completeness, and the primary analytic cohort.
- Previous downstream analysis and machine-learning scripts have been removed from the active workflow.
