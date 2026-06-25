# -*- coding: utf-8 -*-
"""
LASSO Logistic Regression for Binary Classification
Path-based final version

Input folder example:
    C:\\Users\\win\\Desktop\\CHJ\\Lasso\\Larm

Input Excel files:
    Processed_Sensor_Data_Averaged_1min.xlsx
    Processed_Sensor_Data_Averaged_2min.xlsx
    ...
    Processed_Sensor_Data_Averaged_6min.xlsx

Target label:
    Group column
    Cons = 0
    EarlyPDs = 1

Features:
    Excel E:AL columns = pandas iloc[:, 4:38]

Outputs:
    Result Excel file per minute
    ROC curve PNG per minute
    Coefficient barplot PNG per minute
    Bootstrap stability selection per minute
    Master summary Excel file
"""

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

warnings.filterwarnings("ignore")


# ============================================================
# User settings: 여기만 본인 폴더에 맞게 수정하면 됩니다.
# ============================================================

RAW_DATA_DIR = Path(r"C:\Users\win\Desktop\CHJ\Lasso\Larm")
RESULT_DIR = Path(r"C:\Users\win\Desktop\CHJ\Lasso\Larm_Results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FILE_LIST = [
    "Processed_Sensor_Data_Averaged_1min.xlsx",
    "Processed_Sensor_Data_Averaged_2min.xlsx",
    "Processed_Sensor_Data_Averaged_3min.xlsx",
    "Processed_Sensor_Data_Averaged_4min.xlsx",
    "Processed_Sensor_Data_Averaged_5min.xlsx",
    "Processed_Sensor_Data_Averaged_6min.xlsx",
]

GROUP_COL = "Group"
CONTROL_LABEL = "Cons"
PD_LABEL = "EarlyPDs"

# Excel E:AL = pandas index 4:38
FEATURE_START_INDEX = 4
FEATURE_END_INDEX_EXCLUSIVE = 38
EXPECTED_N_FEATURES = 34

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS_CV = 5
CS = np.logspace(-4, 4, 50)

# Bootstrap stability selection
N_BOOTSTRAP = 200
BOOTSTRAP_TEST_SIZE = 0.2
STABILITY_THRESHOLD = 0.50


# ============================================================
# Helper functions
# ============================================================

def check_input_files():
    """Check whether all required Excel files exist."""
    print("\n[Path check]")
    print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"RESULT_DIR  : {RESULT_DIR}")

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"RAW_DATA_DIR does not exist: {RAW_DATA_DIR}")

    files = []
    missing_files = []
    for file_name in FILE_LIST:
        file_path = RAW_DATA_DIR / file_name
        if file_path.exists():
            files.append(file_path)
            print(f"FOUND  : {file_path}")
        else:
            missing_files.append(file_path)
            print(f"MISSING: {file_path}")

    if missing_files:
        missing_text = "\n".join(str(p) for p in missing_files)
        raise FileNotFoundError(
            "Some required Excel files were not found.\n"
            f"Missing files:\n{missing_text}\n\n"
            "Please check RAW_DATA_DIR and exact file names."
        )

    return files


def extract_minute_from_filename(file_path: Path) -> str:
    match = re.search(r"(\d+)min", file_path.name)
    return f"{match.group(1)}min" if match else file_path.stem


def load_and_prepare_data(file_path: Path):
    data = pd.read_excel(file_path, engine="openpyxl")

    # Clean column names
    data.columns = [str(c).strip() for c in data.columns]

    if GROUP_COL not in data.columns:
        raise ValueError(
            f"[{file_path.name}] '{GROUP_COL}' column was not found.\n"
            f"Current columns are:\n{list(data.columns)}"
        )

    # Clean group labels
    data[GROUP_COL] = data[GROUP_COL].astype(str).str.strip()
    data = data[data[GROUP_COL].isin([CONTROL_LABEL, PD_LABEL])].copy()

    if data.empty:
        raise ValueError(
            f"[{file_path.name}] No valid rows after filtering Group labels.\n"
            f"Expected labels: {CONTROL_LABEL}, {PD_LABEL}"
        )

    # E:AL features
    X = data.iloc[:, FEATURE_START_INDEX:FEATURE_END_INDEX_EXCLUSIVE].copy()

    if X.shape[1] != EXPECTED_N_FEATURES:
        print(
            f"WARNING [{file_path.name}]: Expected {EXPECTED_N_FEATURES} features, "
            f"but found {X.shape[1]} features from E:AL."
        )

    # Convert features to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Replace missing values
    X = X.fillna(X.mean(numeric_only=True)).fillna(0)

    # Label mapping
    y = data[GROUP_COL].map({CONTROL_LABEL: 0, PD_LABEL: 1}).astype(int).values
    class_mapping = {CONTROL_LABEL: 0, PD_LABEL: 1}

    if len(np.unique(y)) != 2:
        raise ValueError(
            f"[{file_path.name}] Binary classification requires two classes, "
            f"but found: {np.unique(y)}"
        )

    return X, y, list(X.columns), class_mapping, data


def make_lasso_logistic_pipeline(cv):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegressionCV(
            Cs=CS,
            cv=cv,
            penalty="l1",
            solver="liblinear",
            scoring="roc_auc",
            class_weight=None,
            max_iter=5000,
            random_state=RANDOM_STATE,
            refit=True,
        )),
    ])


def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auc_value = np.nan
    try:
        auc_value = roc_auc_score(y_true, y_prob)
    except Exception:
        pass

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "AUC": auc_value,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def save_roc_curve(y_test, y_prob_test, auc_value, output_png):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    return pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds})


def save_coefficient_barplot(selected_features_df, output_png, title):
    plt.figure(figsize=(9, max(4, 0.35 * max(1, len(selected_features_df)))))

    if selected_features_df.empty:
        plt.text(0.5, 0.5, "No selected features", ha="center", va="center")
        plt.axis("off")
    else:
        plot_df = selected_features_df.sort_values("Coefficient", ascending=True)
        plt.barh(plot_df["Feature"], plot_df["Coefficient"])
        plt.xlabel("Coefficient")
        plt.ylabel("Feature")
        plt.title(title)
        plt.tight_layout()

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()


def bootstrap_stability_selection(X, y, feature_names):
    selection_counts = pd.Series(0, index=feature_names, dtype=float)
    bootstrap_rows = []

    splitter = StratifiedShuffleSplit(
        n_splits=N_BOOTSTRAP,
        test_size=BOOTSTRAP_TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    valid_count = 0

    for i, (train_idx, _) in enumerate(splitter.split(X, y), start=1):
        X_boot = X.iloc[train_idx].copy()
        y_boot = y[train_idx]

        min_class_count = np.bincount(y_boot).min()
        n_splits = min(N_SPLITS_CV, min_class_count)
        if n_splits < 2:
            bootstrap_rows.append({
                "Bootstrap Iteration": i,
                "Best C": np.nan,
                "Equivalent Alpha = 1/C": np.nan,
                "Number of Selected Features": np.nan,
                "Selected Features": "Skipped: too few samples per class",
            })
            continue

        cv_boot = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        try:
            pipe = make_lasso_logistic_pipeline(cv_boot)
            pipe.fit(X_boot, y_boot)

            model = pipe.named_steps["model"]
            coef = model.coef_[0]
            selected_mask = coef != 0
            selected_features = np.array(feature_names)[selected_mask].tolist()

            selection_counts[selected_features] += 1
            valid_count += 1

            bootstrap_rows.append({
                "Bootstrap Iteration": i,
                "Best C": model.C_[0],
                "Equivalent Alpha = 1/C": 1 / model.C_[0],
                "Number of Selected Features": len(selected_features),
                "Selected Features": ", ".join(selected_features) if selected_features else "None",
            })
        except Exception as e:
            bootstrap_rows.append({
                "Bootstrap Iteration": i,
                "Best C": np.nan,
                "Equivalent Alpha = 1/C": np.nan,
                "Number of Selected Features": np.nan,
                "Selected Features": f"Error: {str(e)}",
            })

    valid_n = max(1, valid_count)

    stability_df = pd.DataFrame({
        "Feature": feature_names,
        "Selection Count": selection_counts.values,
        "Selection Frequency": selection_counts.values / valid_n,
        "Stable Selected": (selection_counts.values / valid_n) >= STABILITY_THRESHOLD,
    }).sort_values("Selection Frequency", ascending=False)

    bootstrap_summary_df = pd.DataFrame(bootstrap_rows)

    return stability_df, bootstrap_summary_df


def run_lasso_for_one_file(file_path: Path):
    minute_label = extract_minute_from_filename(file_path)
    print("\n==============================")
    print(f"Running: {file_path.name}")
    print(f"Minute : {minute_label}")
    print("==============================")

    X, y, feature_names, class_mapping, original_data = load_and_prepare_data(file_path)

    minute_output_dir = RESULT_DIR / minute_label
    minute_output_dir.mkdir(parents=True, exist_ok=True)

    min_class_count = np.bincount(y).min()
    n_splits = min(N_SPLITS_CV, min_class_count)

    if n_splits < 2:
        raise ValueError(
            f"[{file_path.name}] Too few samples per class for cross-validation. "
            f"Minimum class count = {min_class_count}"
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipe = make_lasso_logistic_pipeline(cv)
    pipe.fit(X_train, y_train)

    model = pipe.named_steps["model"]
    best_C = model.C_[0]
    best_alpha = 1 / best_C

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)
    y_prob_train = pipe.predict_proba(X_train)[:, 1]
    y_prob_test = pipe.predict_proba(X_test)[:, 1]

    train_metrics = calculate_metrics(y_train, y_pred_train, y_prob_train)
    test_metrics = calculate_metrics(y_test, y_pred_test, y_prob_test)

    coef = model.coef_[0]
    selected_mask = coef != 0
    selected_features = np.array(feature_names)[selected_mask].tolist()

    all_coefficients_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coef,
        "Abs Coefficient": np.abs(coef),
        "Odds Ratio": np.exp(coef),
        "Selected": selected_mask,
    }).sort_values("Abs Coefficient", ascending=False)

    selected_features_df = all_coefficients_df[all_coefficients_df["Selected"]].copy()
    selected_features_df = selected_features_df.sort_values("Odds Ratio", ascending=False)

    odds_ratio_sorted_df = all_coefficients_df.copy().sort_values("Odds Ratio", ascending=False)

    summary_df = pd.DataFrame({
        "Metric": [
            "Minute", "Input File", "Class Mapping", "N Total", "N Cons", "N EarlyPDs",
            "N Features", "Best C", "Equivalent Alpha = 1/C", "Number of Selected Features",
            "Train Accuracy", "Train Sensitivity", "Train Specificity", "Train Precision",
            "Train F1-score", "Train AUC", "Test Accuracy", "Test Sensitivity",
            "Test Specificity", "Test Precision", "Test F1-score", "Test AUC",
        ],
        "Value": [
            minute_label, file_path.name, str(class_mapping), len(y), int(np.sum(y == 0)),
            int(np.sum(y == 1)), X.shape[1], best_C, best_alpha, len(selected_features),
            train_metrics["Accuracy"], train_metrics["Sensitivity"], train_metrics["Specificity"],
            train_metrics["Precision"], train_metrics["F1-score"], train_metrics["AUC"],
            test_metrics["Accuracy"], test_metrics["Sensitivity"], test_metrics["Specificity"],
            test_metrics["Precision"], test_metrics["F1-score"], test_metrics["AUC"],
        ],
    })

    confusion_df = pd.DataFrame({
        "Dataset": ["Train", "Test"],
        "TN": [train_metrics["TN"], test_metrics["TN"]],
        "FP": [train_metrics["FP"], test_metrics["FP"]],
        "FN": [train_metrics["FN"], test_metrics["FN"]],
        "TP": [train_metrics["TP"], test_metrics["TP"]],
    })

    prediction_df = pd.DataFrame({
        "True Label": y_test,
        "True Group": [CONTROL_LABEL if label == 0 else PD_LABEL for label in y_test],
        "Predicted Label": y_pred_test,
        "Predicted Group": [CONTROL_LABEL if label == 0 else PD_LABEL for label in y_pred_test],
        "Predicted Probability EarlyPDs": y_prob_test,
    })

    report_dict = classification_report(
        y_test,
        y_pred_test,
        target_names=[CONTROL_LABEL, PD_LABEL],
        output_dict=True,
        zero_division=0,
    )
    classification_report_df = pd.DataFrame(report_dict).transpose()

    lasso_results_df = pd.DataFrame({
        "Minute": [minute_label],
        "Best C": [best_C],
        "Equivalent Alpha = 1/C": [best_alpha],
        "Number of Selected Features": [len(selected_features)],
        "Selected Features": [", ".join(selected_features) if selected_features else "None"],
    })

    stability_df, bootstrap_summary_df = bootstrap_stability_selection(X, y, feature_names)
    stable_selected_df = stability_df[stability_df["Stable Selected"]].copy()

    roc_png = minute_output_dir / f"[ROC]_{minute_label}.png"
    coef_png = minute_output_dir / f"[Coefficient_Barplot]_{minute_label}.png"

    roc_df = save_roc_curve(y_test, y_prob_test, test_metrics["AUC"], roc_png)
    save_coefficient_barplot(selected_features_df, coef_png, f"LASSO Selected Coefficients - {minute_label}")

    output_excel = minute_output_dir / f"[Result]_LASSO_Logistic_{minute_label}.xlsx"

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        lasso_results_df.to_excel(writer, sheet_name="LASSO Logistic Results", index=False)
        summary_df.to_excel(writer, sheet_name="Performance Summary", index=False)
        selected_features_df.to_excel(writer, sheet_name="Selected Features", index=False)
        odds_ratio_sorted_df.to_excel(writer, sheet_name="Odds Ratio Sorted", index=False)
        all_coefficients_df.to_excel(writer, sheet_name="All Coefficients", index=False)
        stability_df.to_excel(writer, sheet_name="Bootstrap Stability", index=False)
        stable_selected_df.to_excel(writer, sheet_name="Stable Selected Features", index=False)
        bootstrap_summary_df.to_excel(writer, sheet_name="Bootstrap Iterations", index=False)
        confusion_df.to_excel(writer, sheet_name="Confusion Matrix", index=False)
        prediction_df.to_excel(writer, sheet_name="Test Predictions", index=False)
        roc_df.to_excel(writer, sheet_name="ROC Curve Data", index=False)
        classification_report_df.to_excel(writer, sheet_name="Classification Report", index=True)

    print(f"Class mapping: {class_mapping}")
    print(f"Best C: {best_C:.6f}")
    print(f"Alpha = 1/C: {best_alpha:.6f}")
    print(f"Selected features ({len(selected_features)}): {selected_features}")
    print(f"Test AUC: {test_metrics['AUC']:.4f}")
    print(f"Result saved: {output_excel}")

    return {
        "Minute": minute_label,
        "Input File": file_path.name,
        "N Total": len(y),
        "N Cons": int(np.sum(y == 0)),
        "N EarlyPDs": int(np.sum(y == 1)),
        "N Features": X.shape[1],
        "Best C": best_C,
        "Equivalent Alpha = 1/C": best_alpha,
        "Number of Selected Features": len(selected_features),
        "Selected Features": ", ".join(selected_features) if selected_features else "None",
        "Test Accuracy": test_metrics["Accuracy"],
        "Test Sensitivity": test_metrics["Sensitivity"],
        "Test Specificity": test_metrics["Specificity"],
        "Test Precision": test_metrics["Precision"],
        "Test F1-score": test_metrics["F1-score"],
        "Test AUC": test_metrics["AUC"],
        "Result Excel": str(output_excel),
        "ROC PNG": str(roc_png),
        "Coefficient PNG": str(coef_png),
    }


def main():
    files = check_input_files()

    print("\nFiles to analyze:")
    for file_path in files:
        print(f" - {file_path.name}")

    master_rows = []

    for file_path in files:
        try:
            result = run_lasso_for_one_file(file_path)
            master_rows.append(result)
        except Exception as e:
            print(f"ERROR in {file_path.name}: {e}")
            master_rows.append({
                "Minute": extract_minute_from_filename(file_path),
                "Input File": file_path.name,
                "Error": str(e),
            })

    master_summary_df = pd.DataFrame(master_rows)
    master_output = RESULT_DIR / "[Master_Summary]_LASSO_Logistic_1min_to_6min.xlsx"
    master_summary_df.to_excel(master_output, index=False)

    print("\n==============================")
    print("All analyses completed.")
    print(f"Master summary saved: {master_output}")
    print("==============================")


if __name__ == "__main__":
    main()
