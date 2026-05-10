"""[E] Perceptron tự code (binary + multiclass)."""

import numpy as np
import pickle


class Perceptron:
    """
    Single-layer Perceptron tự code.

    Binary mode  (n_classes=2): học luật Perceptron cổ điển
        ŷ = sign(w·x + b),  cập nhật khi ŷ ≠ y

    Multiclass   (n_classes>2): One-vs-Rest — train K binary perceptrons

    Args:
        learning_rate : tốc độ học
        n_epochs      : số epoch
        random_state  : seed
    """

    def __init__(self, learning_rate=0.01, n_epochs=100, random_state=42):
        self.learning_rate = learning_rate
        self.n_epochs      = n_epochs
        self.random_state  = random_state
        self.weights_      = None
        self.bias_         = None
        self.classes_      = None
        self.train_errors_ = []   # số lỗi per epoch

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        n, p = X.shape

        rng            = np.random.RandomState(self.random_state)
        self.classes_  = np.unique(y)
        self.weights_  = rng.randn(p) * 0.01
        self.bias_     = 0.0

        # Binary: map labels to {-1, +1}
        y_bin = np.where(y == self.classes_[-1], 1, -1)

        for _ in range(self.n_epochs):
            errors = 0
            # Shuffle mỗi epoch để tránh oscillation
            idx = rng.permutation(n)
            for i in idx:
                xi = X[i]
                yi = y_bin[i]
                pred = 1 if (np.dot(self.weights_, xi) + self.bias_) >= 0 else -1
                if pred != yi:
                    self.weights_ += self.learning_rate * yi * xi
                    self.bias_    += self.learning_rate * yi
                    errors        += 1
            self.train_errors_.append(errors)
            if errors == 0:
                break
        return self

    def predict(self, X):
        X      = np.asarray(X, dtype=np.float64)
        scores = X @ self.weights_ + self.bias_
        # Map back: ≥0 → positive class, <0 → negative class
        return np.where(scores >= 0, self.classes_[-1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
