import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.datasets import make_moons

"""
# Create a linearly separable dataset with 2 classes, 50 samples each
X, y = make_classification(
    n_samples=100,          # total samples
    n_features=2,           # 2D for easy visualization
    n_redundant=0,          # no redundant features
    n_informative=2,        # both features are informative
    n_clusters_per_class=1, # simple structure
    class_sep=2.0,          # increase separation for linear separability
    random_state=42         # reproducibility
)

# Create DataFrame
df = pd.DataFrame(X, columns=["f1", "f2"])
df["label"] = y

# Save to CSV
file_path = "linear_separable_dataset.csv"
df.to_csv(file_path, index=False)

"""
# Generate a non-linearly separable dataset (moons)
X_nl, y_nl = make_moons(n_samples=100, noise=0.2, random_state=42)

# Convert to DataFrame
df = pd.DataFrame(X_nl, columns=["f1", "f2"])
df["label"] = y_nl

# Save to CSV
file_path_nl = "nonlinear_separable_dataset.csv"
df.to_csv(file_path_nl, index=False)




# Plot the dataset
plt.figure(figsize=(8, 6))
for label in df["label"].unique():
    subset = df[df["label"] == label]
    plt.scatter(subset["f1"], subset["f2"], label=f"Class {label}", alpha=0.7)

plt.title("Linearly Separable Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()