import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression, make_classification, load_digits
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import layers, models, utils

"""
A compact, runnable Python tutorial that demonstrates core ideas of:
- Data science (data loading, exploration)
- Machine learning (regression, classification, clustering)
- Deep learning (simple neural network)

Save as: /c:/Users/kioko/Desktop/4.1/gui git/python.py
Run: python python.py

The script uses synthetic datasets (so no external data needed).
It will try to use TensorFlow for the neural network; if unavailable,
it falls back to scikit-learn's MLPClassifier.

Requires: numpy, pandas, matplotlib, scikit-learn. Optional: tensorflow.
"""


# Visualization (optional headless-safe)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, classification_report, confusion_matrix
)

# Try to import TensorFlow (optional)
try:
    HAS_TF = True
except Exception:
    HAS_TF = False

def show_plot():
    if HAS_MPL:
        plt.show()
    else:
        print("(matplotlib not available - skipping plots)")

def regression_demo():
    print("\n=== Regression demo (Linear Regression) ===")
    # Generate synthetic regression data
    X, y = make_regression(n_samples=200, n_features=1, noise=20.0, random_state=1)
    df = pd.DataFrame({"x": X.ravel(), "y": y})
    print("Data sample:\n", df.head())

    # Quick EDA
    print("Descriptive stats:\n", df.describe())

    # Train/test split and scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Fit linear regression
    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    print("R^2:", round(r2_score(y_test, y_pred), 3))
    print("RMSE:", round(mean_squared_error(y_test, y_pred, squared=False), 3))

    if HAS_MPL:
        plt.figure()
        plt.scatter(X, y, label="data", alpha=0.6)
        xs = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        xs_s = scaler.transform(xs)
        plt.plot(xs, model.predict(xs_s), color="red", label="linear fit")
        plt.title("Linear Regression (synthetic)")
        plt.legend()
        show_plot()

def classification_demo():
    print("\n=== Classification demo (Logistic Regression) ===")
    # 2D classification toy dataset to visualize decision boundary
    X, y = make_classification(
        n_samples=300, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, class_sep=1.2, random_state=2
    )
    df = pd.DataFrame(X, columns=["f1", "f2"])
    df["label"] = y
    print("Data sample:\n", df.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression()
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    if HAS_MPL:
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_s = scaler.transform(grid)
        Z = clf.predict(grid_s).reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.2)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", alpha=0.6)
        plt.title("Logistic Regression decision boundary")
        show_plot()

def clustering_demo():
    print("\n=== Clustering demo (KMeans) ===")
    # Use digits dataset projected to 2D via PCA for illustration
    digits = load_digits()
    X = digits.data
    y = digits.target
    # simple PCA-like projection using SVD for 2 components
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X2 = U[:, :2] * S[:2]

    kmeans = KMeans(n_clusters=10, random_state=0)
    labels = kmeans.fit_predict(X2)
    print("KMeans cluster counts:", np.bincount(labels))

    if HAS_MPL:
        plt.figure(figsize=(6, 4))
        plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", s=15)
        plt.title("KMeans clusters on digits (2D projection)")
        show_plot()

def deep_learning_demo():
    print("\n=== Deep learning demo (Simple neural network on digits) ===")
    digits = load_digits()
    X = digits.data / 16.0  # normalize 0..1
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if HAS_TF:
        print("Using TensorFlow/Keras (if available).")
        num_classes = len(np.unique(y))
        model = models.Sequential([
            layers.Input(shape=(X.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        # Train briefly (small epochs for demo)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print("Test accuracy (Keras):", round(acc, 3))
    else:
        print("TensorFlow not found — falling back to scikit-learn's MLPClassifier.")
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300, random_state=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Test accuracy (sklearn MLP):", round(accuracy_score(y_test, y_pred), 3))
        print("Classification report:\n", classification_report(y_test, y_pred))

def tips():
    print("\n=== Quick learning tips ===")
    tips = [
        "1) Data science = data collection -> cleaning -> exploration -> modeling -> communication.",
        "2) Machine learning = algorithms to learn patterns (supervised, unsupervised, reinforcement).",
        "3) Deep learning = neural networks with many layers; useful for images, audio, text.",
        "4) Practice: try small datasets, read metrics, visualize results, iterate.",
        "5) Tools: numpy, pandas, scikit-learn for ML; TensorFlow/PyTorch for deep learning."
    ]
    for t in tips:
        print(t)

def main():
    print("GitHub Copilot: running a compact ML/AI/Deep Learning & Data Science demo script.")
    regression_demo()
    classification_demo()
    clustering_demo()
    deep_learning_demo()
    tips()
    print("\nDone. Explore the code and modify parameters to learn more!")

if __name__ == "__main__":
    main()