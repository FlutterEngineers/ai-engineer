import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
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

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Classes:", data.target_names)


# --------------------------------------------------
# 2. Split the data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 3. Configure cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# --------------------------------------------------
# 4. Define the models
# --------------------------------------------------

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            random_state=42,
        ),
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(random_state=42),
    ),
}


# --------------------------------------------------
# 5. Train and compare models
# --------------------------------------------------

results = []

# Store scores where higher values mean stronger
# evidence that a sample is malignant.
malignant_scores = {}

# Save the fitted Logistic Regression model for
# the threshold experiment later.
logistic_model = None

# Original dataset:
# 0 = malignant
# 1 = benign
#
# For our evaluation, convert malignant to the
# positive class (1) and benign to the negative class (0).
y_malignant = (y_test == 0).astype(int)

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Cross-validation using training data only.
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    # Fit the model using all training samples.
    model.fit(X_train, y_train)

    # Save the fitted model.
    if name == "Logistic Regression":
        logistic_model = model

    # Predict the original dataset classes.
    predictions = model.predict(X_test)

    # Obtain scores for malignant detection.
    if name == "SVM":
        # Higher negative decision scores indicate
        # stronger evidence for malignant class 0.
        scores = -model.decision_function(X_test)
    else:
        # Column 0 corresponds to malignant.
        scores = model.predict_proba(X_test)[:, 0]

    malignant_scores[name] = scores

    # Evaluate model performance.
    test_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    test_auc = roc_auc_score(
        y_malignant,
        scores,
    )

    average_precision = average_precision_score(
        y_malignant,
        scores,
    )

    results.append({
        "Model": name,
        "CV Accuracy": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "Test Accuracy": test_accuracy,
        "ROC-AUC (Malignant)": test_auc,
        "Average Precision": average_precision,
    })

    print(f"CV accuracy: {cv_scores.mean():.4f}")
    print(f"CV standard deviation: {cv_scores.std():.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Malignant ROC-AUC: {test_auc:.4f}")
    print(f"Malignant Average Precision: {average_precision:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=data.target_names,
            zero_division=0,
        )
    )


# --------------------------------------------------
# 6. Compare final model results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV Accuracy",
    ascending=False,
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))


# --------------------------------------------------
# 7. Plot ROC curves
# --------------------------------------------------

plt.figure(figsize=(9, 6))

for name, scores in malignant_scores.items():

    fpr, tpr, thresholds = roc_curve(
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

# Random-classifier reference line.
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


# --------------------------------------------------
# 8. Plot Precision-Recall curves
# --------------------------------------------------

plt.figure(figsize=(9, 6))

for name, scores in malignant_scores.items():

    precision, recall, thresholds = precision_recall_curve(
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

# Baseline is the proportion of malignant samples.
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


# --------------------------------------------------
# 9. Compare Logistic Regression thresholds
# --------------------------------------------------

print("\n========== THRESHOLD COMPARISON ==========")

# Get probabilities for malignant class 0.
malignant_probability = logistic_model.predict_proba(
    X_test
)[:, 0]

# Test selected thresholds.
thresholds_to_test = [0.10, 0.20, 0.30, 0.40, 0.50]

threshold_results = []

for threshold in thresholds_to_test:

    # Predict malignant when its probability meets
    # or exceeds the selected threshold.
    predicted_malignant = (
        malignant_probability >= threshold
    ).astype(int)

    # Actual malignant is positive class 1 here.
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


# --------------------------------------------------
# 10. Plot Precision and Recall vs Threshold
# --------------------------------------------------

# Evaluate many thresholds to see how the trade-off changes.
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

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

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


# --------------------------------------------------
# 11. Final summary
# --------------------------------------------------

print("\n========== FINAL SUMMARY ==========")

best_model = results_df.iloc[0]

print("Best model by cross-validation accuracy:")
print(best_model["Model"])

print(f"CV accuracy: {best_model['CV Accuracy']:.4f}")
print(f"Test accuracy: {best_model['Test Accuracy']:.4f}")
print(f"Malignant ROC-AUC: {best_model['ROC-AUC (Malignant)']:.4f}")
print(f"Malignant Average Precision: {best_model['Average Precision']:.4f}")

print("\nAll model comparisons and plots completed.")
