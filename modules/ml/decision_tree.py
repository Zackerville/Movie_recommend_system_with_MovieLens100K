"""[E] Decision Tree Regressor tự code (variance reduction split)."""

import numpy as np
import pickle

class _Node:
    __slots__ = ('feature', 'threshold', 'value', 'left', 'right')

    def __init__(self, feature=None, threshold=None, value=None,
                 left=None, right=None):
        self.feature   = feature
        self.threshold = threshold
        self.value     = value     # leaf node value = mean(y)
        self.left      = left
        self.right     = right

    @property
    def is_leaf(self):
        return self.value is not None

class DecisionTreeRegressor:
    """Regression tree tự code — variance reduction split, leaf = mean(y)."""

    def __init__(self, max_depth=10, min_samples_split=20, min_samples_leaf=10,
                 max_features=None, n_thresholds=25, random_state=42):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.n_thresholds      = n_thresholds
        self.random_state      = random_state
        self.root_             = None
        self.n_features_       = None
        self._rng              = np.random.RandomState(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.n_features_ = X.shape[1]
        self.root_ = self._build(X, y, depth=0)
        return self

    def _n_try(self):
        n = self.n_features_
        if self.max_features is None:     return n
        if self.max_features == 'sqrt':   return max(1, int(np.sqrt(n)))
        if self.max_features == 'log2':   return max(1, int(np.log2(n)))
        return min(n, int(self.max_features))

    def _best_split(self, X, y):
        n        = len(y)
        var_y    = float(np.var(y))
        best_vr  = 0.0
        best_f   = None
        best_thr = None

        feat_idx = self._rng.choice(self.n_features_, size=self._n_try(), replace=False)

        for f in feat_idx:
            col   = X[:, f]
            pcts  = np.percentile(col, np.linspace(5, 95, self.n_thresholds))
            thrs  = np.unique(pcts)

            for thr in thrs:
                lm = col <= thr
                rm = ~lm
                nl = lm.sum()
                nr = rm.sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                vl  = float(np.var(y[lm])) if nl > 1 else 0.0
                vr  = float(np.var(y[rm])) if nr > 1 else 0.0
                vr_ = var_y - (nl / n) * vl - (nr / n) * vr
                if vr_ > best_vr:
                    best_vr  = vr_
                    best_f   = f
                    best_thr = float(thr)

        return best_f, best_thr

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                float(np.var(y)) < 1e-7):
            return _Node(value=float(np.mean(y)))

        f, thr = self._best_split(X, y)
        if f is None:
            return _Node(value=float(np.mean(y)))

        lm = X[:, f] <= thr
        rm = ~lm
        node       = _Node(feature=f, threshold=thr)
        node.left  = self._build(X[lm], y[lm], depth + 1)
        node.right = self._build(X[rm], y[rm], depth + 1)
        return node

    def predict(self, X):
        X   = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float32)
        for i, x in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                node = node.left if x[node.feature] <= node.threshold else node.right
            out[i] = node.value
        return out

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
