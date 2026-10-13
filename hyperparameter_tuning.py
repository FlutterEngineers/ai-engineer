# ============================================================
# HYPERPARAMETER TUNING AND MODEL EVALUATION
# Breast Cancer Classification
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Classes:", data.target_names)


# ============================================================
# 2. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 3. CONFIGURE CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ============================================================
# 4. DEFINE MODELS AND HYPERPARAMETER GRIDS
# ============================================================

model_configs = {
    "Logistic Regression": {
        "model": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                random_state=42,
            ),
        ),
        "params": {
            "logisticregression__C": [0.01, 0.1, 1, 10, 100],
        },
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            random_state=42,
            n_jobs=1,
        ),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2],
            "min_samples_split": [2, 5],
        },
    },

    "SVM": {
        "model": make_pipeline(
            StandardScaler(),
            SVC(
                probability=True,
                random_state=42,
            ),
        ),
        "params": {
            "svc__C": [0.1, 1, 10],
            "svc__kernel": ["linear", "rbf"],
            "svc__gamma": ["scale", "auto"],
        },
    },
}


# ============================================================
# 5. TUNE, TRAIN, AND EVALUATE MODELS
# ============================================================

results = []

# This dictionary stores the fitted best estimator for each model.
tuned_models = {}

# Scores where higher values indicate stronger evidence of malignancy.
malignant_scores = {}

# Dataset labels: 0 = malignant, 1 = benign.
# Convert to 1 = malignant, 0 = benign for malignant-specific metrics.
y_malignant = (y_test == 0).astype(int)

for name, config in model_configs.items():

    print("\n" + "=" * 55)
    print(f"Tuning: {name}")
    print("=" * 55)

    search = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    # Fit and tune using training data only.
    search.fit(X_train, y_train)

    # Save the best fitted estimator.
    best_estimator = search.best_estimator_
    tuned_models[name] = best_estimator

    print("\nBest parameters:")
    print(search.best_params_)

    print(f"\nBest CV accuracy: {search.best_score_:.4f}")

    # Predict the original dataset classes.
    predictions = best_estimator.predict(X_test)

    # Get scores for malignant detection.
    if hasattr(best_estimator, "predict_proba"):
        malignant_probability = best_estimator.predict_proba(
            X_test
        )[:, 0]

    else:
        # For binary SVC without probabilities, invert the
        # decision scores because sklearn orders classes as
        # malignant (0), then benign (1).
        decision_scores = best_estimator.decision_function(X_test)
        malignant_probability = -decision_scores

    malignant_scores[name] = malignant_probability

    test_accuracy = accuracy_score(y_test, predictions)

    malignant_auc = roc_auc_score(
        y_malignant,
        malignant_probability,
    )

    malignant_ap = average_precision_score(
        y_malignant,
        malignant_probability,
    )

    results.append({
        "Model": name,
        "Best CV Accuracy": search.best_score_,
        "Test Accuracy": test_accuracy,
        "Malignant ROC-AUC": malignant_auc,
        "Malignant Average Precision": malignant_ap,
    })

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Malignant ROC-AUC: {malignant_auc:.4f}")
    print(f"Malignant Average Precision: {malignant_ap:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=data.target_names,
            zero_division=0,
        )
    )


# ============================================================
# 6. COMPARE TUNED MODELS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Best CV Accuracy",
    ascending=False,
).reset_index(drop=True)

print("\n========== TUNED MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

print("\n========== BEST PARAMETERS ==========")

for name, config in model_configs.items():
    # Retrieve the fitted best estimator's parameters.
    print(f"\n{name}:")
    print({
        key: tuned_models[name].get_params()[key]
        for key in config["params"]
    })


# ============================================================
# 7. PLOT ROC CURVES
# ============================================================

plt.figure(figsize=(9, 6))

for name, scores in malignant_scores.items():

    fpr, tpr, _ = roc_curve(
        y_malignant,
        scores,
    )

    auc = roc_auc_score(
        y_malignant,
        scores,
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.4f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random classifier",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Malignant Recall)")
plt.title("ROC Curves - Malignant Detection")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 8. PLOT PRECISION-RECALL CURVES
# ============================================================

plt.figure(figsize=(9, 6))

for name, scores in malignant_scores.items():

    precision, recall, _ = precision_recall_curve(
        y_malignant,
        scores,
    )

    ap = average_precision_score(
        y_malignant,
        scores,
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP = {ap:.4f})",
    )

baseline = y_malignant.mean()

plt.axhline(
    y=baseline,
    linestyle="--",
    color="gray",
    label=f"Baseline ({baseline:.3f})",
)

plt.xlabel("Recall (Malignant)")
plt.ylabel("Precision (Malignant)")
plt.title("Precision-Recall Curves - Malignant Detection")
plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 9. LOGISTIC REGRESSION THRESHOLD COMPARISON
# ============================================================

print("\n========== THRESHOLD COMPARISON ==========")

# Use the tuned Logistic Regression estimator.
logistic_model = tuned_models["Logistic Regression"]

# Probability of malignant class 0.
malignant_probability = logistic_model.predict_proba(
    X_test
)[:, 0]

thresholds_to_test = [0.10, 0.20, 0.30, 0.40, 0.50]

threshold_results = []

for threshold in thresholds_to_test:

    predicted_malignant = (
        malignant_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_malignant,
        predicted_malignant,
        labels=[0, 1],
    ).ravel()

    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0

    threshold_results.append({
        "Threshold": threshold,
        "Malignant Recall": recall,
        "Malignant Precision": precision,
        "Missed Malignant": fn,
        "False Alarms": fp,
    })

threshold_df = pd.DataFrame(threshold_results)

print(threshold_df.round(4).to_string(index=False))


# ============================================================
# 10. PLOT PRECISION AND RECALL VS THRESHOLD
# ============================================================

threshold_values = np.linspace(0.01, 0.99, 99)

precision_values = []
recall_values = []

for threshold in threshold_values:

    predicted_malignant = (
        malignant_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_malignant,
        predicted_malignant,
        labels=[0, 1],
    ).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    precision_values.append(precision)
    recall_values.append(recall)

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_values,
    precision_values,
    label="Precision",
    linewidth=2,
)

plt.plot(
    threshold_values,
    recall_values,
    label="Recall",
    linewidth=2,
)

plt.xlabel("Malignant Probability Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Classification Threshold")
plt.xlim(0, 1)
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

best_result = results_df.iloc[0]
best_model_name = best_result["Model"]
best_model = tuned_models[best_model_name]

print("\n========== FINAL SUMMARY ==========")
print("Best model by cross-validation accuracy:", best_model_name)
print(f"Best CV accuracy: {best_result['Best CV Accuracy']:.4f}")
print(f"Test accuracy: {best_result['Test Accuracy']:.4f}")
print(f"Malignant ROC-AUC: {best_result['Malignant ROC-AUC']:.4f}")
print(
    "Malignant Average Precision: "
    f"{best_result['Malignant Average Precision']:.4f}"
)


# ============================================================
# 12. SAVE BEST TUNED MODEL
# ============================================================

model_artifact = {
    "model": best_model,
    "model_name": best_model_name,
    "feature_names": list(data.feature_names),
    "target_names": list(data.target_names),
    "positive_class": "malignant",
    "best_cv_accuracy": float(best_result["Best CV Accuracy"]),
    "test_accuracy": float(best_result["Test Accuracy"]),
    "malignant_roc_auc": float(best_result["Malignant ROC-AUC"]),
}

# Save beside this Python script.
model_path = Path(__file__).resolve().parent / "best_model.joblib"

joblib.dump(model_artifact, model_path)

print("\n========== MODEL SAVED ==========")
print("Model:", best_model_name)
print("File:", model_path)
print("Number of input features:", len(data.feature_names))

print("\nHyperparameter tuning and model saving completed.")