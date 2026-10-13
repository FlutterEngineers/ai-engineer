
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the saved model
artifact = joblib.load("best_model.joblib")

# Load a real sample from the dataset
dataset = load_breast_cancer()
sample = pd.DataFrame(
    dataset.data[0:1],
    columns=artifact["feature_names"]
)

# Make a prediction
prediction = int(artifact["model"].predict(sample)[0])
predicted_label = artifact["target_names"][prediction]
actual_label = dataset.target_names[dataset.target[0]]

print("Model:", artifact["model_name"])
print("Number of features:", sample.shape[1])
print("Predicted class:", predicted_label)
print("Actual dataset class:", actual_label)
print("Prediction correct:", predicted_label == actual_label)
