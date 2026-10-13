
import joblib
import pandas as pd

# 1. Load the saved model
artifact = joblib.load("best_model.joblib")

model = artifact["model"]
feature_names = artifact["feature_names"]
target_names = artifact["target_names"]

print("Model loaded:", artifact["model_name"])
print("Expected features:", len(feature_names))

# 2. Ask the user for all 30 feature values
values = []

print("\nEnter the 30 feature values.")
print("Use numeric values from a valid dataset sample for testing.\n")

for feature in feature_names:
    while True:
        try:
            value = float(input(f"{feature}: "))

            if not pd.notna(value) or value < 0:
                print("Please enter a finite, non-negative number.")
                continue

            values.append(value)
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

# 3. Create input with the correct feature names and order
sample = pd.DataFrame([values], columns=feature_names)

# 4. Generate the prediction
prediction = int(model.predict(sample)[0])

print("\n========== PREDICTION ==========")
print("Predicted class:", target_names[prediction])

# 5. Display model probabilities when available
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(sample)[0]

    print("\nModel-estimated class probabilities:")
    for index, probability in enumerate(probabilities):
        print(f"{target_names[index]}: {probability:.2%}")
