# ============================================================
# KNHANES Data Cleaning Script
# Depression Severity Modeling
# ============================================================

library(tidyverse)
library(readr)

# ------------------------------------------------------------
# 1. Load merged KNHANES data
# ------------------------------------------------------------

DATA_PATH <- "KNHANES_even_years_2014_2024_merged.csv"

raw_df <- read_csv(DATA_PATH, show_col_types = FALSE)

cat("Original data shape:", dim(raw_df), "\n")


# ------------------------------------------------------------
# 2. Define required variables
# ------------------------------------------------------------

required_vars <- c(
  
  # ID / survey year
  "year",
  "ID",
  
  # Outcome: PHQ-9
  "mh_PHQ_S",
  
  # Demographics
  "age",
  "sex",
  
  # Socioeconomic covariates
  "ho_incm",
  "edu",
  "marri_1",
  "EC1_1",
  
  # Stress perception
  "BP1",
  
  # Sleep variables
  "BP8",
  "BP16_1",
  "BP16_2",
  "BP16_11",
  "BP16_12",
  "BP16_13",
  "BP16_14",
  "BP16_21",
  "BP16_22",
  "BP16_23",
  "BP16_24",
  
  # Physical activity variables
  "BE3_71", "BE3_72", "BE3_73", "BE3_74",
  "BE3_75", "BE3_76", "BE3_77", "BE3_78",
  "BE3_81", "BE3_82", "BE3_83", "BE3_84",
  "BE3_85", "BE3_86", "BE3_87", "BE3_88",
  "BE3_91", "BE3_92", "BE3_93", "BE3_94",
  
  # Resting pulse rate variables
  "HE_PLS",
  "HE_PLS_30",
  "HE_mPLS",
  
  # Body / behavior covariates
  "HE_BMI",
  "HE_obe",
  "BS1_1",
  "BS3_1",
  "BD1",
  "BD1_11",
  
  # Disease history
  "DI1_dg",    # 고혈압
  "DI3_dg",    # 뇌졸중
  "DI5_dg",    # 심근경색증
  "DI6_dg"     # 협심증
)


existing_vars <- required_vars[required_vars %in% names(raw_df)]
missing_vars <- setdiff(required_vars, names(raw_df))

cat("\nNumber of required variables:", length(required_vars), "\n")
cat("Existing variables:", length(existing_vars), "\n")
cat("Missing variables:", length(missing_vars), "\n")

if (length(missing_vars) > 0) {
  cat("\nMissing variables:\n")
  print(missing_vars)
}

df_cohort <- raw_df %>%
  select(all_of(existing_vars))

cat("\ndf_cohort initial shape:", dim(df_cohort), "\n")


# ------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------

to_num <- function(x) {
  suppressWarnings(as.numeric(str_trim(as.character(x))))
}

valid_range <- function(x, lower, upper) {
  x <- to_num(x)
  if_else(x >= lower & x <= upper, x, NA_real_)
}

valid_values <- function(x, values) {
  x <- to_num(x)
  if_else(x %in% values, x, NA_real_)
}

remove_missing_codes <- function(x, missing_codes = c(88, 99)) {
  x <- to_num(x)
  if_else(x %in% missing_codes, NA_real_, x)
}


# ============================================================
# Exclusion criteria
# ============================================================


# ------------------------------------------------------------
# 4. PHQ-9 total score and 2-level severity group
# ------------------------------------------------------------

df_cohort <- df_cohort %>%
  mutate(
    mh_PHQ_S = valid_range(mh_PHQ_S, 0, 27),
    
    PHQ_group = case_when(
      mh_PHQ_S >= 0 & mh_PHQ_S <= 4  ~ "Minimal",
      mh_PHQ_S >= 5 & mh_PHQ_S <= 27 ~ "Mild_or_greater",
      TRUE ~ NA_character_
    ),
    
    PHQ_group = factor(
      PHQ_group,
      levels = c("Minimal", "Mild_or_greater")
    )
  ) %>%
  drop_na(PHQ_group)

cat("\nPHQ-9 check after dropping missing:\n")
print(summary(df_cohort$mh_PHQ_S))
print(table(df_cohort$PHQ_group, useNA = "ifany"))
cat("df_cohort shape after PHQ drop:", dim(df_cohort), "\n")

# 44313 -> 32294
# n = 12019 excluded


# ------------------------------------------------------------
# 5. Age
# ------------------------------------------------------------

df_cohort <- df_cohort %>%
  mutate(
    age_numeric = valid_range(age, 0, 120)
  ) %>%
  filter(
    !is.na(age_numeric),
    age_numeric >= 19
  ) %>%
  select(-any_of("age"))

cat("\nAge check after adult restriction:\n")
print(summary(df_cohort$age_numeric))
cat("df_cohort shape after age cleaning:", dim(df_cohort), "\n")

# 32294 -> 32294
# n = 0 excluded

# ------------------------------------------------------------
# 6. Resting pulse rate
# ------------------------------------------------------------
# HE_mPLS   = 60-second pulse
# HE_PLS_30 = 30-second pulse -> x2
# HE_PLS    = 15-second pulse -> x4
#
# Priority:
# 1. HE_mPLS
# 2. HE_PLS_30 * 2
# 3. HE_PLS * 4
#
# Exclusion:
# - pulse missing
# - pulse < 40 bpm
# - pulse > 200 bpm
#
# 각 단계에서 다음을 계산한다.
# - 제거 환자 수
# - 제외 후 남은 총 환자 수
# - 제외 후 남은 우울 증상군(PHQ-9 >= 5) 수
# - 제외 후 전체 대상자 중 우울 증상군 비율
# ------------------------------------------------------------

df_cohort <- df_cohort %>%
  mutate(
    
    HE_mPLS_numeric = if ("HE_mPLS" %in% names(.)) {
      to_num(HE_mPLS)
    } else {
      NA_real_
    },
    
    HE_PLS_30_numeric = if ("HE_PLS_30" %in% names(.)) {
      to_num(HE_PLS_30)
    } else {
      NA_real_
    },
    
    HE_PLS_numeric = if ("HE_PLS" %in% names(.)) {
      to_num(HE_PLS)
    } else {
      NA_real_
    },
    
    pulse_rate_60 = case_when(
      !is.na(HE_mPLS_numeric) ~ HE_mPLS_numeric,
      !is.na(HE_PLS_30_numeric) ~ HE_PLS_30_numeric * 2,
      !is.na(HE_PLS_numeric) ~ HE_PLS_numeric * 4,
      TRUE ~ NA_real_
    ),
    
    pulse_abnormal = !is.na(pulse_rate_60) &
      (pulse_rate_60 < 40 | pulse_rate_60 > 200)
  )


cat("\nPulse rate check:\n")
print(summary(df_cohort$pulse_rate_60))
print(table(df_cohort$pulse_abnormal, useNA = "ifany"))


cat("\nPulse rate missing/abnormal counts by year:\n")

print(
  df_cohort %>%
    group_by(year) %>%
    summarise(
      n = n(),
      pulse_missing = sum(is.na(pulse_rate_60)),
      pulse_abnormal_n = sum(pulse_abnormal, na.rm = TRUE),
      .groups = "drop"
    )
)


# ------------------------------------------------------------
# 6-1. 안정시 심박수 제외 적용
# ------------------------------------------------------------

n_before_pulse <- nrow(df_cohort)

df_cohort <- df_cohort %>%
  filter(
    !is.na(pulse_rate_60),
    pulse_abnormal == FALSE
  )

n_pulse_excluded <- n_before_pulse - nrow(df_cohort)
n_pulse_remaining <- nrow(df_cohort)

n_pulse_depression <- sum(
  df_cohort$PHQ_group == "Mild_or_greater",
  na.rm = TRUE
)

pct_pulse_depression <- round(
  n_pulse_depression /
    n_pulse_remaining * 100,
  1
)


cat("\n========================================\n")
cat("Resting pulse exclusion summary\n")
cat("========================================\n")

cat("Excluded:", n_pulse_excluded, "\n")
cat("Remaining:", n_pulse_remaining, "\n")
cat(
  "Depression:",
  n_pulse_depression,
  "(",
  pct_pulse_depression,
  "%)\n"
)


# 원래 심박수 변수 제거
df_cohort <- df_cohort %>%
  select(
    -any_of(c(
      "HE_mPLS",
      "HE_PLS_30",
      "HE_PLS",
      "HE_mPLS_numeric",
      "HE_PLS_30_numeric",
      "HE_PLS_numeric"
    ))
  )

# Expected:
# 32294 -> 32093
# n = 201 excluded



# ------------------------------------------------------------
# 7. Cardiovascular disease exclusions
# ------------------------------------------------------------
# DI1_dg = 고혈압 의사진단 여부
# DI3_dg = 뇌졸중 의사진단 여부
# DI5_dg = 심근경색증 의사진단 여부
# DI6_dg = 협심증 의사진단 여부
#
# 0 = 없음
# 1 = 있음
#
# 각 질환이 있다고 응답한 대상자(=1)를 순차적으로 제외한다.
# 모름 / 무응답 / 해당 연도에 변수가 없는 경우(NA)는 유지한다.
#
# 각 단계에서 다음을 계산한다.
# - 제거 환자 수
# - 제외 후 남은 총 환자 수
# - 제외 후 남은 우울 증상군(PHQ-9 >= 5) 수
# - 제외 후 전체 대상자 중 우울 증상군 비율
# ------------------------------------------------------------


# ------------------------------------------------------------
# 7-1. 질환 변수 정리
# ------------------------------------------------------------

df_cohort <- df_cohort %>%
  mutate(
    hypertension_dx = valid_values(DI1_dg, c(0, 1)),
    stroke_dx       = valid_values(DI3_dg, c(0, 1)),
    mi_dx           = valid_values(DI5_dg, c(0, 1)),
    angina_dx       = valid_values(DI6_dg, c(0, 1))
  )


cat("\nCardiovascular disease variables check:\n")

cat("\nHypertension:\n")
print(table(df_cohort$hypertension_dx, useNA = "ifany"))

cat("\nStroke:\n")
print(table(df_cohort$stroke_dx, useNA = "ifany"))

cat("\nMyocardial infarction:\n")
print(table(df_cohort$mi_dx, useNA = "ifany"))

cat("\nAngina:\n")
print(table(df_cohort$angina_dx, useNA = "ifany"))



# ------------------------------------------------------------
# 질환별 제외 결과 저장용 표
# ------------------------------------------------------------

cvd_exclusion_summary <- tibble(
  exclusion_criterion = character(),
  excluded_n = integer(),
  remaining_n = integer(),
  depression_n = integer(),
  depression_pct = numeric()
)


# ------------------------------------------------------------
# 질환별 제외 함수
# ------------------------------------------------------------

record_exclusion <- function(data_before,
                             disease_var,
                             criterion) {
  
  n_before <- nrow(data_before)
  
  data_after <- data_before %>%
    filter(
      {{ disease_var }} != 1 |
        is.na({{ disease_var }})
    )
  
  excluded_n <- n_before - nrow(data_after)
  remaining_n <- nrow(data_after)
  
  depression_n <- sum(
    data_after$PHQ_group == "Mild_or_greater",
    na.rm = TRUE
  )
  
  depression_pct <- round(
    depression_n /
      remaining_n * 100,
    1
  )
  
  summary_row <- tibble(
    exclusion_criterion = criterion,
    excluded_n = excluded_n,
    remaining_n = remaining_n,
    depression_n = depression_n,
    depression_pct = depression_pct
  )
  
  list(
    data = data_after,
    summary = summary_row
  )
}



# ------------------------------------------------------------
# 7-2. 고혈압 제외
# ------------------------------------------------------------

result <- record_exclusion(
  df_cohort,
  hypertension_dx,
  "Hypertension"
)

df_cohort <- result$data

cvd_exclusion_summary <- bind_rows(
  cvd_exclusion_summary,
  result$summary
)



# ------------------------------------------------------------
# 7-3. 뇌졸중 제외
# ------------------------------------------------------------

result <- record_exclusion(
  df_cohort,
  stroke_dx,
  "Stroke"
)

df_cohort <- result$data

cvd_exclusion_summary <- bind_rows(
  cvd_exclusion_summary,
  result$summary
)



# ------------------------------------------------------------
# 7-4. 심근경색 제외
# ------------------------------------------------------------

result <- record_exclusion(
  df_cohort,
  mi_dx,
  "Myocardial infarction"
)

df_cohort <- result$data

cvd_exclusion_summary <- bind_rows(
  cvd_exclusion_summary,
  result$summary
)



# ------------------------------------------------------------
# 7-5. 협심증 제외
# ------------------------------------------------------------

result <- record_exclusion(
  df_cohort,
  angina_dx,
  "Angina"
)

df_cohort <- result$data

cvd_exclusion_summary <- bind_rows(
  cvd_exclusion_summary,
  result$summary
)



# ------------------------------------------------------------
# 7-6. 심혈관계 질환 제외 결과
# ------------------------------------------------------------

cvd_exclusion_summary <- cvd_exclusion_summary %>%
  mutate(
    depression_n_pct = paste0(
      depression_n,
      " (",
      depression_pct,
      ")"
    )
  )


cat("\n========================================\n")
cat("Cardiovascular disease exclusion summary\n")
cat("========================================\n")

print(cvd_exclusion_summary)


cat(
  "\nTotal excluded due to cardiovascular disease:",
  sum(cvd_exclusion_summary$excluded_n),
  "\n"
)

cat(
  "Final cohort size:",
  nrow(df_cohort),
  "\n"
)


cat("\nN by year after cardiovascular disease exclusions:\n")

print(
  table(
    df_cohort$year,
    useNA = "ifany"
  )
)


# ------------------------------------------------------------
# 원자료 질환 변수 제거
# ------------------------------------------------------------

df_cohort <- df_cohort %>%
  select(
    -any_of(c(
      "DI1_dg",
      "DI3_dg",
      "DI5_dg",
      "DI6_dg"
    ))
  )

# Expected:
# 32093 -> 23418
# Total n = 8675 excluded

# ------------------------------------------------------------
# 8. Stress perception: BP1
# ------------------------------------------------------------
# Original:
# 1 = 매우 많이 느낌
# 2 = 많이 느낌
# 3 = 조금 느낌
# 4 = 거의 느끼지 않음
#
# Reverse:
# Higher value = greater perceived stress

df_cohort <- df_cohort %>%
  mutate(
    
    BP1_numeric = valid_values(
      BP1,
      c(1, 2, 3, 4)
    ),
    
    BP1_reversed = case_when(
      BP1_numeric == 1 ~ 4,
      BP1_numeric == 2 ~ 3,
      BP1_numeric == 3 ~ 2,
      BP1_numeric == 4 ~ 1,
      TRUE ~ NA_real_
    )
  )

cat("\nBP1 reversed check:\n")
print(table(df_cohort$BP1_reversed, useNA = "ifany"))

df_cohort <- df_cohort %>%
  select(
    -any_of(c(
      "BP1",
      "BP1_numeric"
    ))
  )


# ------------------------------------------------------------
# 9. Physical activity
# ------------------------------------------------------------
# Weekly physical activity is converted to MET-min/week.
#
# Vigorous activity:
# 8 MET × minutes/week
#
# Moderate activity:
# 4 MET × minutes/week
#
# Transport activity:
# 4 MET × minutes/week
#
# Final group:
# <600 MET-min/week
# >=600 MET-min/week


clean_activity_domain <- function(
    data,
    participation_col,
    days_col,
    hours_col,
    minutes_col,
    output_col
) {
  
  p_clean <- paste0(participation_col, "_clean")
  d_clean <- paste0(days_col, "_clean")
  h_clean <- paste0(hours_col, "_clean")
  m_clean <- paste0(minutes_col, "_clean")
  
  data %>%
    mutate(
      
      "{p_clean}" :=
        valid_values(
          .data[[participation_col]],
          c(1, 2)
        ),
      
      "{d_clean}" :=
        valid_range(
          .data[[days_col]],
          1,
          7
        ),
      
      "{h_clean}" :=
        valid_range(
          .data[[hours_col]],
          0,
          24
        ),
      
      "{m_clean}" :=
        valid_range(
          .data[[minutes_col]],
          0,
          59
        ),
      
      daily_minutes_temp =
        .data[[h_clean]] * 60 +
        .data[[m_clean]],
      
      "{output_col}" := case_when(
        
        .data[[p_clean]] == 2 ~ 0,
        
        .data[[p_clean]] == 1 &
          !is.na(.data[[d_clean]]) &
          !is.na(daily_minutes_temp) &
          daily_minutes_temp <= 1440 ~
          .data[[d_clean]] * daily_minutes_temp,
        
        TRUE ~ NA_real_
      )
    ) %>%
    select(-daily_minutes_temp)
}


df_cohort <- df_cohort %>%
  
  clean_activity_domain(
    "BE3_71",
    "BE3_72",
    "BE3_73",
    "BE3_74",
    "vigorous_work_min_week"
  ) %>%
  
  clean_activity_domain(
    "BE3_75",
    "BE3_76",
    "BE3_77",
    "BE3_78",
    "vigorous_leisure_min_week"
  ) %>%
  
  clean_activity_domain(
    "BE3_81",
    "BE3_82",
    "BE3_83",
    "BE3_84",
    "moderate_work_min_week"
  ) %>%
  
  clean_activity_domain(
    "BE3_85",
    "BE3_86",
    "BE3_87",
    "BE3_88",
    "moderate_leisure_min_week"
  ) %>%
  
  clean_activity_domain(
    "BE3_91",
    "BE3_92",
    "BE3_93",
    "BE3_94",
    "transport_min_week"
  ) %>%
  
  mutate(
    
    # -----------------------------
    # MET-min/week
    # -----------------------------
    
    total_met_min_week = if_else(
      
      !is.na(vigorous_work_min_week) &
        !is.na(vigorous_leisure_min_week) &
        !is.na(moderate_work_min_week) &
        !is.na(moderate_leisure_min_week) &
        !is.na(transport_min_week),
      
      8 * vigorous_work_min_week +
        8 * vigorous_leisure_min_week +
        4 * moderate_work_min_week +
        4 * moderate_leisure_min_week +
        4 * transport_min_week,
      
      NA_real_
    ),
    
    # -----------------------------
    # 600 MET-min/week cutoff
    # -----------------------------
    
    physical_activity_group = case_when(
      
      is.na(total_met_min_week) ~ NA_character_,
      
      total_met_min_week < 600 ~
        "<600 MET-min/week",
      
      total_met_min_week >= 600 ~
        ">=600 MET-min/week"
    ),
    
    physical_activity_group = factor(
      physical_activity_group,
      levels = c(
        "<600 MET-min/week",
        ">=600 MET-min/week"
      )
    )
  )


cat("\nPhysical activity MET check:\n")
print(summary(df_cohort$total_met_min_week))

cat(
  "Missing total_met_min_week:",
  sum(is.na(df_cohort$total_met_min_week)),
  "\n"
)

cat("\nPhysical activity group:\n")
print(
  table(
    df_cohort$physical_activity_group,
    useNA = "ifany"
  )
)


df_cohort <- df_cohort %>%
  select(
    -any_of(c(
      
      "BE3_71", "BE3_72", "BE3_73", "BE3_74",
      "BE3_75", "BE3_76", "BE3_77", "BE3_78",
      "BE3_81", "BE3_82", "BE3_83", "BE3_84",
      "BE3_85", "BE3_86", "BE3_87", "BE3_88",
      "BE3_91", "BE3_92", "BE3_93", "BE3_94",
      
      "BE3_71_clean", "BE3_72_clean",
      "BE3_73_clean", "BE3_74_clean",
      
      "BE3_75_clean", "BE3_76_clean",
      "BE3_77_clean", "BE3_78_clean",
      
      "BE3_81_clean", "BE3_82_clean",
      "BE3_83_clean", "BE3_84_clean",
      
      "BE3_85_clean", "BE3_86_clean",
      "BE3_87_clean", "BE3_88_clean",
      
      "BE3_91_clean", "BE3_92_clean",
      "BE3_93_clean", "BE3_94_clean",
      
      "vigorous_work_min_week",
      "vigorous_leisure_min_week",
      "moderate_work_min_week",
      "moderate_leisure_min_week",
      "transport_min_week"
    ))
  )


# ------------------------------------------------------------
# 10. Sleep duration
# ------------------------------------------------------------
# 2014:
# BP8 = average sleep hours
#
# 2020 / 2022:
# BP16_1 = weekday sleep
# BP16_2 = weekend sleep
#
# 2016 / 2018 / 2024:
# clock-based bedtime/wake time
#
# Weighted average:
# (weekday × 5 + weekend × 2) / 7


average_sleep_years <- c(2014)

direct_sleep_years <- c(
  2020,
  2022
)

clock_sleep_years <- c(
  2016,
  2018,
  2024
)

all_sleep_years <- c(
  2014,
  2016,
  2018,
  2020,
  2022,
  2024
)


clock_total_minutes <- function(hour, minute) {
  
  hour <- remove_missing_codes(hour)
  minute <- remove_missing_codes(minute)
  
  valid_time <-
    hour >= 1 &
    hour <= 24 &
    minute >= 0 &
    minute <= 59
  
  hour <- if_else(
    valid_time,
    hour,
    NA_real_
  )
  
  minute <- if_else(
    valid_time,
    minute,
    NA_real_
  )
  
  hour <- if_else(
    hour == 24,
    0,
    hour
  )
  
  hour * 60 + minute
}


sleep_duration_from_clock <- function(
    bed_hour,
    bed_minute,
    wake_hour,
    wake_minute
) {
  
  bed_total <-
    clock_total_minutes(
      bed_hour,
      bed_minute
    )
  
  wake_total <-
    clock_total_minutes(
      wake_hour,
      wake_minute
    )
  
  duration <-
    wake_total - bed_total
  
  duration <- if_else(
    wake_total > bed_total,
    duration,
    wake_total + 1440 - bed_total
  )
  
  duration <- if_else(
    !is.na(bed_total) &
      !is.na(wake_total),
    duration,
    NA_real_
  )
  
  duration / 60
}


df_cohort <- df_cohort %>%
  mutate(
    weekday_sleep_hours = NA_real_,
    weekend_sleep_hours = NA_real_
  )


# Direct sleep years: 2020 / 2022
df_cohort <- df_cohort %>%
  mutate(
    
    weekday_sleep_hours = if_else(
      year %in% direct_sleep_years,
      remove_missing_codes(BP16_1),
      weekday_sleep_hours
    ),
    
    weekend_sleep_hours = if_else(
      year %in% direct_sleep_years,
      remove_missing_codes(BP16_2),
      weekend_sleep_hours
    )
  )


# Clock sleep years: 2016 / 2018 / 2024
df_cohort <- df_cohort %>%
  mutate(
    
    weekday_sleep_clock =
      sleep_duration_from_clock(
        BP16_11,
        BP16_12,
        BP16_13,
        BP16_14
      ),
    
    weekend_sleep_clock =
      sleep_duration_from_clock(
        BP16_21,
        BP16_22,
        BP16_23,
        BP16_24
      ),
    
    weekday_sleep_hours = if_else(
      year %in% clock_sleep_years,
      weekday_sleep_clock,
      weekday_sleep_hours
    ),
    
    weekend_sleep_hours = if_else(
      year %in% clock_sleep_years,
      weekend_sleep_clock,
      weekend_sleep_hours
    )
  )


df_cohort <- df_cohort %>%
  mutate(
    
    sleep_avg_weighted =
      (
        weekday_sleep_hours * 5 +
          weekend_sleep_hours * 2
      ) / 7,
    
    # 2014
    sleep_avg_weighted = if_else(
      year %in% average_sleep_years,
      remove_missing_codes(BP8),
      sleep_avg_weighted
    ),
    
    # Flag only
    sleep_implausible =
      !is.na(sleep_avg_weighted) &
      (
        sleep_avg_weighted < 3 |
          sleep_avg_weighted > 14
      )
  )


cat("\nSleep duration check:\n")

print(
  summary(
    df_cohort$sleep_avg_weighted
  )
)

print(
  table(
    df_cohort$sleep_implausible,
    useNA = "ifany"
  )
)


cat("\nSleep duration missing counts by year:\n")

print(
  df_cohort %>%
    group_by(year) %>%
    summarise(
      n = n(),
      sleep_missing =
        sum(
          is.na(
            sleep_avg_weighted
          )
        ),
      sleep_implausible_n =
        sum(
          sleep_implausible,
          na.rm = TRUE
        ),
      .groups = "drop"
    )
)


df_cohort <- df_cohort %>%
  select(
    -any_of(c(
      
      "BP8",
      
      "BP16_1",
      "BP16_2",
      
      "BP16_11",
      "BP16_12",
      "BP16_13",
      "BP16_14",
      
      "BP16_21",
      "BP16_22",
      "BP16_23",
      "BP16_24",
      
      "weekday_sleep_hours",
      "weekend_sleep_hours",
      
      "weekday_sleep_clock",
      "weekend_sleep_clock"
    ))
  )


# ------------------------------------------------------------
# 11. Demographic / socioeconomic / behavior covariates
# ------------------------------------------------------------

df_cohort <- df_cohort %>%
  mutate(
    
    # Sex
    # 1 = Male
    # 2 = Female
    sex_numeric =
      valid_values(
        sex,
        c(1, 2)
      ),
    
    # Household income
    # 1 = Low
    # 2 = Lower-middle
    # 3 = Upper-middle
    # 4 = High
    household_income =
      valid_values(
        ho_incm,
        c(1, 2, 3, 4)
      ),
    
    # Education
    # 1 = Elementary or less
    # 2 = Middle school
    # 3 = High school
    # 4 = College or above
    edu_numeric =
      valid_values(
        edu,
        c(1, 2, 3, 4)
      ),
    
    # Marital status
    # 1 = Married
    # 2 = Unmarried
    marri_1_numeric =
      valid_values(
        marri_1,
        c(1, 2)
      ),

    # Working status
    # 1 = Yes
    # 2 = No
    employment_status =
      valid_values(
        EC1_1,
        c(1, 2)
      ),

    # Body mass index
    BMI_numeric =
      valid_range(
        HE_BMI,
        10,
        80
      ),

    BMI_group = case_when(
      BMI_numeric < 23 ~ "<23 kg/m2",
      BMI_numeric >= 23 & BMI_numeric < 25 ~ "23-24.9 kg/m2",
      BMI_numeric >= 25 ~ ">=25 kg/m2",
      TRUE ~ NA_character_
    ),

    BMI_group = factor(
      BMI_group,
      levels = c(
        "<23 kg/m2",
        "23-24.9 kg/m2",
        ">=25 kg/m2"
      )
    ),
    
    # Obesity category
    HE_obe_numeric =
      valid_values(
        HE_obe,
        c(1, 2, 3, 4, 5, 6)
      ),
    
    # Smoking status
    # 1 = Never
    # 2 = Previous
    # 3 = Current
    smoking_history = case_when(
      
      valid_values(
        BS1_1,
        c(1, 2, 3)
      ) == 3 ~ 1,
      
      valid_values(
        BS3_1,
        c(1, 2, 3)
      ) == 3 ~ 2,
      
      valid_values(
        BS3_1,
        c(1, 2, 3)
      ) %in% c(1, 2) ~ 3,
      
      TRUE ~ NA_real_
    ),
    
    # Alcohol drinking status
    # 1 = Never
    # 2 = Previous
    # 3 = Current
    BD1_11_numeric =
      case_when(
        valid_values(BD1, c(1, 2)) == 1 ~ 1,
        valid_values(BD1, c(1, 2)) == 2 &
          valid_values(BD1_11, c(1, 2, 3, 4, 5, 6)) == 1 ~ 2,
        valid_values(BD1, c(1, 2)) == 2 &
          valid_values(BD1_11, c(1, 2, 3, 4, 5, 6)) %in% c(2, 3, 4, 5, 6) ~ 3,
        TRUE ~ NA_real_
      )
  ) %>%
  select(
    -any_of(c(
      "sex",
      "ho_incm",
      "edu",
      "marri_1",
      "EC1_1",
      "HE_BMI",
      "HE_obe",
      "BS1_1",
      "BS3_1",
      "BD1",
      "BD1_11"
    ))
  )


cat("\nCovariates check:\n")

print(
  table(
    df_cohort$sex_numeric,
    useNA = "ifany"
  )
)

print(
  table(
    df_cohort$household_income,
    useNA = "ifany"
  )
)

print(
  table(
    df_cohort$edu_numeric,
    useNA = "ifany"
  )
)

print(
  table(
    df_cohort$marri_1_numeric,
    useNA = "ifany"
  )
)

print(
  table(
    df_cohort$employment_status,
    useNA = "ifany"
  )
)

print(summary(df_cohort$BMI_numeric))

print(
  table(
    df_cohort$BMI_group,
    useNA = "ifany"
  )
)

print(
  table(
    df_cohort$HE_obe_numeric,
    useNA = "ifany"
  )
)

print(
  table(
    df_cohort$smoking_history,
    useNA = "ifany"
  )
)

print(
  table(
    df_cohort$BD1_11_numeric,
    useNA = "ifany"
  )
)


# ------------------------------------------------------------
# 12. Remove temporary variables
# ------------------------------------------------------------

df_cohort <- df_cohort %>%
  select(
    -any_of(c(
      "BS3_1",
      "pulse_abnormal",
      "hypertension_dx",
      "stroke_dx",
      "mi_dx",
      "angina_dx",
      "cvd_exclusion"
    ))
  )


# ------------------------------------------------------------
# 13. Final variable order
# ------------------------------------------------------------

df_cohort <- df_cohort %>%
  select(
    any_of(c(
      
      "year",
      "ID",
      
      # Outcome
      "mh_PHQ_S",
      "PHQ_group",
      
      # Main predictor
      "pulse_rate_60",
      
      # Behavioral / lifestyle predictors
      "BP1_reversed",
      "total_met_min_week",
      "physical_activity_group",
      "sleep_avg_weighted",
      "sleep_implausible",
      
      # Demographics / covariates
      "age_numeric",
      "sex_numeric",
      "household_income",
      "edu_numeric",
      "marri_1_numeric",
      "employment_status",
      "BMI_numeric",
      "BMI_group",
      "HE_obe_numeric",
      "smoking_history",
      "BD1_11_numeric"
    ))
  )


cat("\nFinal variables kept:\n")
print(names(df_cohort))

cat(
  "\nFinal df_cohort shape before predictor missing-value handling:",
  dim(df_cohort),
  "\n"
)


# ------------------------------------------------------------
# 14. Final checks
# ------------------------------------------------------------

cat("\nPHQ group:\n")
print(
  table(
    df_cohort$PHQ_group,
    useNA = "ifany"
  )
)

cat("\nPhysical activity group:\n")
print(
  table(
    df_cohort$physical_activity_group,
    useNA = "ifany"
  )
)

cat("\nMissing values:\n")

print(
  colSums(
    is.na(df_cohort)
  )
)


# ------------------------------------------------------------
# 15. Save cleaned cohort
# ------------------------------------------------------------

write_csv(
  df_cohort,
  "df_cohort_cleaned.csv"
)

cat(
  "\nSaved file: df_cohort_cleaned.csv\n"
)


print(table(df_cohort$PHQ_group, useNA = "ifany"))
