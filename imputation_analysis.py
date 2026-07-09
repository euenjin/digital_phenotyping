from contextlib import redirect_stdout
from pathlib import Path
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
RESULT_DIR = OUTPUT_DIR / "analysis_result"
RESULT_DIR.mkdir(exist_ok=True)
IMPUTATION_DIR = RESULT_DIR / "imputation"
IMPUTATION_ADJUSTED_DIR = IMPUTATION_DIR / "adjusted"
IMPUTATION_UNADJUSTED_DIR = IMPUTATION_DIR / "unadjusted"
IMPUTATION_COMPARISON_DIR = IMPUTATION_DIR / "comparison"
IMPUTATION_LOG_DIR = IMPUTATION_DIR / "logs"
for path in [
    IMPUTATION_ADJUSTED_DIR,
    IMPUTATION_UNADJUSTED_DIR,
    IMPUTATION_COMPARISON_DIR,
    IMPUTATION_LOG_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
CLEANING_SCRIPT = OUTPUT_DIR / "1_data_cleaning.py"

TARGET_SOURCE = "PHQ_sum"
TARGET = "depression_phq10"
EXCLUDED_PREDICTORS = ["BP1_numeric"]

ADJUSTMENT_COVARIATES = [
    "sex_numeric",
    "age_numeric",
    "edu_numeric",
    "marri_1_numeric",
    "HE_obe_numeric",
    "BS1_1_recoded",
    "BD2_31_numeric",
]

PREDICTOR_COLUMNS = [
    "BP1_reversed",
    "pulse_irregular",
    "total_mvpa_min_week",
    "weekday_sleep_hours",
    "weekend_sleep_hours",
    "sleep_avg_weighted",
    "HE_glu_numeric",
    "HE_sbp_numeric",
    "HE_dbp_numeric",
    "pulse_pressure",
    "MAP",
    "sex_numeric",
    "age_numeric",
    "edu_numeric",
    "marri_1_numeric",
    "HE_obe_numeric",
    "BS1_1_recoded",
    "BD2_31_numeric",
]
UNADJUSTED_PREDICTOR_COLUMNS = [
    col for col in PREDICTOR_COLUMNS if col not in ADJUSTMENT_COVARIATES
]

CONTINUOUS_COLUMNS = [
    "total_mvpa_min_week",
    "weekday_sleep_hours",
    "weekend_sleep_hours",
    "sleep_avg_weighted",
    "HE_glu_numeric",
    "HE_sbp_numeric",
    "HE_dbp_numeric",
    "pulse_pressure",
    "MAP",
    "age_numeric",
]
CATEGORICAL_COLUMNS = [
    "BP1_reversed",
    "pulse_irregular",
    "sex_numeric",
    "edu_numeric",
    "marri_1_numeric",
    "HE_obe_numeric",
    "BS1_1_recoded",
    "BD2_31_numeric",
]


def require_modeling_dependencies():
    missing = []
    try:
        import numpy as np  # noqa: F401
    except ModuleNotFoundError:
        missing.append("numpy")
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError:
        missing.append("scikit-learn")
    try:
        import shap  # noqa: F401
    except ModuleNotFoundError:
        missing.append("shap")

    if missing:
        raise SystemExit(
            "Missing required package(s): "
            + ", ".join(missing)
            + "\nInstall them with: python -m pip install "
            + " ".join(missing)
        )


def load_cleaned_dataframe():
    log_path = IMPUTATION_LOG_DIR / "cleaning_output_log.txt"
    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file):
            namespace = runpy.run_path(str(CLEANING_SCRIPT), run_name="__imputation_cleaning__")
    df_clean = namespace.get("df_clean")
    if df_clean is None:
        raise RuntimeError("1_data_cleaning.py did not create df_clean.")
    return df_clean


def prepare_target_nonmissing_data(df_clean, predictor_columns, excluded_predictors):
    missing_columns = [
        col for col in [TARGET_SOURCE, *predictor_columns] if col not in df_clean.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns in df_clean: {missing_columns}")

    model_df = df_clean[[TARGET_SOURCE, *predictor_columns]].copy()
    for col in model_df.columns:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    before_target_drop = len(model_df)
    model_df = model_df.dropna(subset=[TARGET_SOURCE])
    after_target_drop = len(model_df)
    model_df[TARGET] = (model_df[TARGET_SOURCE] >= 10).astype(int)

    X = model_df[predictor_columns]
    y = model_df[TARGET]
    missingness = (
        X.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    missingness["missing_rate"] = missingness["missing_count"] / len(X)
    missingness["imputation_strategy"] = missingness["feature"].apply(
        lambda col: "median" if col in CONTINUOUS_COLUMNS else "most_frequent"
    )

    info = {
        "rows_before_target_drop": before_target_drop,
        "rows_after_target_drop": after_target_drop,
        "rows_dropped_missing_target": before_target_drop - after_target_drop,
        "positive_cases": int(y.sum()),
        "negative_cases": int((1 - y).sum()),
        "positive_rate": float(y.mean()),
        "excluded_predictors": ", ".join(excluded_predictors),
    }
    return model_df, X, y, missingness, info


def build_preprocessor(predictor_columns):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    continuous = [col for col in predictor_columns if col in CONTINUOUS_COLUMNS]
    categorical = [col for col in predictor_columns if col in CATEGORICAL_COLUMNS]

    return ColumnTransformer(
        transformers=[
            ("continuous", SimpleImputer(strategy="median"), continuous),
            ("categorical", SimpleImputer(strategy="most_frequent"), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(predictor_columns):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return [
        (
            "Logistic Regression",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(predictor_columns)),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "Random Forest",
            Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(predictor_columns)),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=500,
                            max_depth=None,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
    ]


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
        "f1": f1_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fitted_model": model,
    }


def clean_feature_names(preprocessor, predictor_columns):
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        continuous = [col for col in predictor_columns if col in CONTINUOUS_COLUMNS]
        categorical = [col for col in predictor_columns if col in CATEGORICAL_COLUMNS]
        return [*continuous, *categorical]


def shap_values_for_positive_class(explainer, X):
    import numpy as np

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        return shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def explain_best_model(best_result, X_train, X_test, predictor_columns, output_prefix):
    import numpy as np
    import shap

    best_name = best_result["model"]
    fitted_model = best_result["fitted_model"]
    preprocessor = fitted_model.named_steps["preprocess"]
    feature_names = clean_feature_names(preprocessor, predictor_columns)
    shap_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)

    if best_name == "Logistic Regression":
        scaler = fitted_model.named_steps["scaler"]
        classifier = fitted_model.named_steps["classifier"]
        transformed_train = pd.DataFrame(
            scaler.transform(preprocessor.transform(X_train)),
            columns=feature_names,
            index=X_train.index,
        )
        transformed_sample = pd.DataFrame(
            scaler.transform(preprocessor.transform(shap_sample)),
            columns=feature_names,
            index=shap_sample.index,
        )
        explainer = shap.LinearExplainer(classifier, transformed_train)
        shap_values = shap_values_for_positive_class(explainer, transformed_sample)
        shap_plot_data = transformed_sample
    else:
        classifier = fitted_model.named_steps["classifier"]
        transformed_sample = pd.DataFrame(
            preprocessor.transform(shap_sample),
            columns=feature_names,
            index=shap_sample.index,
        )
        explainer = shap.TreeExplainer(classifier)
        shap_values = shap_values_for_positive_class(explainer, transformed_sample)
        shap_plot_data = transformed_sample

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": shap_plot_data.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(
        output_prefix / "best_model_shap_importance.csv",
        index=False,
    )

    plt.figure()
    shap.summary_plot(shap_values, shap_plot_data, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(
        output_prefix / "best_model_shap_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, shap_plot_data, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(
        output_prefix / "best_model_shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return importance


def plot_imputed_adjusted_vs_unadjusted(results_df):
    metrics = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    best_by_variant = (
        results_df.sort_values(["variant", "roc_auc", "f1"], ascending=[True, False, False])
        .groupby("variant", as_index=False)
        .first()
    )
    chart_df = best_by_variant.melt(
        id_vars=["variant", "model"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    chart_df.pivot(index="metric", columns="variant", values="score").loc[metrics].plot(
        kind="bar",
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Imputed Model Performance: Adjusted vs Unadjusted")
    ax.legend(title="Model set")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(IMPUTATION_COMPARISON_DIR / "adjusted_vs_unadjusted_model_performance.png", dpi=300)
    plt.close()


def plot_complete_case_vs_imputed(imputed_results_df):
    complete_case_path = (
        RESULT_DIR
        / "complete_case"
        / "comparison"
        / "adjusted_vs_unadjusted_model_performance.csv"
    )
    if not complete_case_path.exists():
        return None

    complete_case_df = pd.read_csv(complete_case_path)
    complete_case_df["analysis"] = "Complete-case"
    imputed_results_df = imputed_results_df.copy()
    imputed_results_df["analysis"] = "Imputed"
    combined = pd.concat([complete_case_df, imputed_results_df], ignore_index=True)

    best_by_group = (
        combined.sort_values(["analysis", "variant", "roc_auc", "f1"])
        .groupby(["analysis", "variant"], as_index=False)
        .tail(1)
    )
    best_by_group["label"] = best_by_group["analysis"] + " " + best_by_group["variant"]

    metrics = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    chart_df = best_by_group.melt(
        id_vars=["label", "model"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    chart_df.pivot(index="metric", columns="label", values="score").loc[metrics].plot(
        kind="bar",
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Complete-case vs Imputed Best Model Performance")
    ax.legend(title="Analysis set", fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = IMPUTATION_COMPARISON_DIR / "complete_case_vs_imputed_model_performance.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def run_imputation_variant(
    variant,
    df_clean,
    predictor_columns,
    excluded_predictors,
    output_prefix,
):
    from sklearn.model_selection import train_test_split

    model_df, X, y, missingness, info = prepare_target_nonmissing_data(
        df_clean,
        predictor_columns,
        excluded_predictors,
    )
    model_df.to_csv(
        output_prefix / "modeling_dataset_target_nonmissing.csv",
        index=False,
    )
    missingness.to_csv(output_prefix / "missingness_for_imputation.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = [
        evaluate_model(name, model, X_train, X_test, y_train, y_test)
        for name, model in build_models(predictor_columns)
    ]
    results_df = pd.DataFrame(
        [{k: v for k, v in result.items() if k != "fitted_model"} for result in results]
    ).sort_values(["roc_auc", "f1"], ascending=False)
    results_df.insert(0, "variant", variant)
    results_df.to_csv(output_prefix / "model_performance.csv", index=False)

    best_result = max(results, key=lambda result: (result["roc_auc"], result["f1"]))
    shap_importance = explain_best_model(
        best_result,
        X_train,
        X_test,
        predictor_columns,
        output_prefix,
    )

    summary_lines = [
        f"{variant} imputed model set",
        f"Predictors: {', '.join(predictor_columns)}",
        f"Excluded predictors: {info['excluded_predictors']}",
        "Imputation: median for continuous predictors, most_frequent for categorical predictors",
        f"Rows before target drop: {info['rows_before_target_drop']}",
        f"Rows after target drop: {info['rows_after_target_drop']}",
        f"Rows dropped for missing target: {info['rows_dropped_missing_target']}",
        f"Positive cases: {info['positive_cases']}",
        f"Negative cases: {info['negative_cases']}",
        f"Positive rate: {info['positive_rate']:.4f}",
        "",
        "Model performance:",
        results_df.to_string(index=False),
        "",
        f"Best model by ROC-AUC then F1: {best_result['model']}",
        "",
        "Top SHAP features:",
        shap_importance.head(15).to_string(index=False),
        "",
        "Most-missing imputed predictors:",
        missingness.sort_values("missing_rate", ascending=False).head(10).to_string(index=False),
    ]
    return {
        "variant": variant,
        "results_df": results_df,
        "summary_lines": summary_lines,
    }


def main():
    require_modeling_dependencies()

    df_clean = load_cleaned_dataframe()
    adjusted = run_imputation_variant(
        "Adjusted",
        df_clean,
        PREDICTOR_COLUMNS,
        EXCLUDED_PREDICTORS,
        IMPUTATION_ADJUSTED_DIR,
    )
    unadjusted = run_imputation_variant(
        "Unadjusted",
        df_clean,
        UNADJUSTED_PREDICTOR_COLUMNS,
        [*EXCLUDED_PREDICTORS, *ADJUSTMENT_COVARIATES],
        IMPUTATION_UNADJUSTED_DIR,
    )

    imputed_results_df = pd.concat(
        [adjusted["results_df"], unadjusted["results_df"]],
        ignore_index=True,
    )
    imputed_results_df.to_csv(IMPUTATION_COMPARISON_DIR / "model_performance.csv", index=False)

    plot_imputed_adjusted_vs_unadjusted(imputed_results_df)
    complete_case_chart = plot_complete_case_vs_imputed(imputed_results_df)

    summary_lines = [
        "PHQ-9 depression prediction with predictor imputation",
        f"Target: {TARGET}=1 when {TARGET_SOURCE} >= 10",
        "Target was not imputed; rows with missing target were excluded.",
        "",
        *adjusted["summary_lines"],
        "",
        *unadjusted["summary_lines"],
        "",
        "Saved charts:",
        "analysis_result/imputation/comparison/adjusted_vs_unadjusted_model_performance.png",
    ]
    if complete_case_chart is not None:
        summary_lines.append(
            "analysis_result/imputation/comparison/complete_case_vs_imputed_model_performance.png"
        )

    summary = "\n".join(summary_lines)
    (IMPUTATION_DIR / "analysis_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
