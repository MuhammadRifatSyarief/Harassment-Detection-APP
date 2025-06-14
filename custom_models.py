import numpy as np
from collections import Counter
import time
import pandas as pd # Needed for y.values check

# ==============================================================================
# --- Helper Functions & Classes for Custom Models (from pasted_content.txt) --=
# ==============================================================================

def calculate_gini(y):
    """
    Calculates the Gini impurity for a set of labels.

    Args:
        y (np.ndarray): A NumPy array of labels.

    Returns:
        float: The Gini impurity score. Returns 0 if the input array is empty.
    """
    if y.size == 0:
        return 0 # No impurity in an empty set

    # Count occurrences of each class
    counts = np.bincount(y)
    # Calculate probabilities
    probabilities = counts / y.size
    # Calculate Gini impurity
    gini = 1.0 - np.sum(probabilities**2)
    return gini

def find_best_split(X, y, n_features_subset=None):
    """
    Finds the best feature and threshold to split the data based on Gini impurity reduction.

    Args:
        X (np.ndarray): Feature matrix for the current node. Shape (n_samples, n_features).
        y (np.ndarray): Labels for the current node. Shape (n_samples,).
        n_features_subset (int, optional): The number of features to randomly consider at this split.
                                          If None, all features are considered (standard Decision Tree).
                                          If specified, used for Random Forest feature randomness.

    Returns:
        tuple: (best_feature_idx, best_threshold, best_gini)
               - best_feature_idx: Index of the feature to split on.
               - best_threshold: Threshold value for the split.
               - best_gini: The minimum weighted Gini impurity achieved by this split.
               Returns (None, None, float(inf)) if no valid split is found.
    """
    n_samples, n_features_total = X.shape
    if n_samples <= 1:
        return None, None, float('inf') # Cannot split further

    best_gini = float('inf')
    best_feature_idx = None
    best_threshold = None

    # Determine which features to consider
    if n_features_subset is not None and n_features_subset < n_features_total:
        feature_indices = np.random.choice(n_features_total, n_features_subset, replace=False)
    else:
        feature_indices = range(n_features_total)

    for feature_idx in feature_indices:
        feature_values = X[:, feature_idx]
        unique_thresholds = np.unique(feature_values)

        if len(unique_thresholds) > 1:
            potential_thresholds = (unique_thresholds[:-1] + unique_thresholds[1:]) / 2
        else:
            potential_thresholds = unique_thresholds

        for threshold in potential_thresholds:
            left_mask = feature_values <= threshold
            right_mask = feature_values > threshold

            y_left, y_right = y[left_mask], y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gini_left = calculate_gini(y_left)
            gini_right = calculate_gini(y_right)
            weighted_gini = (len(y_left) / n_samples) * gini_left + (len(y_right) / n_samples) * gini_right

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature_idx = feature_idx
                best_threshold = threshold

    return best_feature_idx, best_threshold, best_gini

# Modifikasi kelas Node untuk menyimpan distribusi kelas di daun
class Node:
    """Represents a node in the Decision Tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None, class_distribution=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Prediction for leaf node (majority class)
        self.class_distribution = class_distribution  # Store class distribution for probability calculation

    def is_leaf_node(self):
        return self.value is not None

# Modifikasi DecisionTree untuk mendukung predict_proba
class DecisionTree:
    """Decision Tree Classifier implemented from scratch."""
    def __init__(self, max_depth=10, min_samples_split=10, n_features_subset=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_subset = n_features_subset
        self.root = None
        self.n_classes_ = None  # Will store number of classes


    def _most_common_label(self, y):
        if y.size == 0:
            return 0 # Default label for empty set
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        # Handle case where y might be empty or have only one class
        if n_samples == 0:
            # Or perhaps return a default node based on parent? For now, simple leaf.
            return Node(value=0) # Or some other default
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        best_feature_idx, best_threshold, best_gini = find_best_split(X, y, self.n_features_subset)

        if best_gini == float('inf') or best_feature_idx is None:
             leaf_value = self._most_common_label(y)
             return Node(value=leaf_value)

        # Create child nodes recursively
        left_mask = X[:, best_feature_idx] <= best_threshold
        right_mask = X[:, best_feature_idx] > best_threshold

        # Check for empty splits which shouldn't happen if find_best_split worked correctly
        if not np.any(left_mask) or not np.any(right_mask):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_child = self._build_tree(X[left_mask, :], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask, :], y[right_mask], depth + 1)

        return Node(feature_idx=best_feature_idx, threshold=best_threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        # print(f"Fitting Decision Tree (max_depth={self.max_depth}, min_samples_split={self.min_samples_split})...")
        # start_time = time.time()
        y_np = y.values if isinstance(y, pd.Series) else np.array(y)
        X_np = X.toarray() if hasattr(X, "toarray") else np.array(X)

        if self.n_features_subset is not None and self.n_features_subset > X_np.shape[1]:
            # print(f"Warning: n_features_subset ({self.n_features_subset}) > actual features ({X_np.shape[1]}). Using all features.")
            self.n_features_subset = None

        self.root = self._build_tree(X_np, y_np)
        # end_time = time.time()
        # print(f"Decision Tree fitting completed in {end_time - start_time:.2f} seconds.")

    def _traverse_tree(self, x, node):
        if node is None: # Should not happen with proper build, but safety check
             return 0 # Or some default prediction
        if node.is_leaf_node():
            return node.value

        # Handle case where feature index might be out of bounds if X structure changed
        if node.feature_idx >= len(x):
             # This indicates an issue, maybe return majority class or handle error
             # print(f"Warning: Feature index {node.feature_idx} out of bounds for input sample.")
             return 0 # Default prediction

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        if self.root is None:
            raise Exception("Decision Tree must be fitted before prediction.")
        X_np = X.toarray() if hasattr(X, "toarray") else np.array(X)
        predictions = np.array([self._traverse_tree(x, self.root) for x in X_np])
        return predictions

# Note: The RandomForestGPU class using cuML is omitted as the error message
# referred to a standard 'RandomForest' class, implying a CPU-based one was saved.
# We will define a CPU-based RandomForest using the DecisionTree above.

class RandomForest:
    """Random Forest Classifier implemented from scratch using custom DecisionTree."""
    def __init__(self, n_trees=30, max_depth=10, min_samples_split=20, n_features_subset=None, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_subset = n_features_subset # Number of features for each tree
        self.random_state = random_state
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        # Set seed for reproducibility if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        # Generate indices with replacement
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        # print(f"Fitting Random Forest (n_trees={self.n_trees})...")
        # start_time = time.time()
        self.trees = []
        y_np = y.values if isinstance(y, pd.Series) else np.array(y)
        X_np = X.toarray() if hasattr(X, "toarray") else np.array(X)

        # Determine the number of features to use per tree if not specified
        if self.n_features_subset is None:
            self.n_features_subset = int(np.sqrt(X_np.shape[1]))
        elif isinstance(self.n_features_subset, str) and self.n_features_subset.lower() == 'all':
             self.n_features_subset = X_np.shape[1]
        elif isinstance(self.n_features_subset, int) and self.n_features_subset > X_np.shape[1]:
             # print(f"Warning: n_features_subset ({self.n_features_subset}) > actual features ({X_np.shape[1]}). Using all features.")
             self.n_features_subset = X_np.shape[1]

        for i in range(self.n_trees):
            # print(f"  Fitting tree {i+1}/{self.n_trees}...")
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features_subset=self.n_features_subset # Pass feature subset size here
            )
            X_sample, y_sample = self._bootstrap_sample(X_np, y_np)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        # end_time = time.time()
        # print(f"Random Forest fitting completed in {end_time - start_time:.2f} seconds.")

    def predict(self, X):
        if not self.trees:
            raise Exception("Random Forest must be fitted before prediction.")
        X_np = X.toarray() if hasattr(X, "toarray") else np.array(X)

        # Make predictions with each tree
        tree_preds = np.array([tree.predict(X_np) for tree in self.trees])
        # Majority vote
        # tree_preds shape: (n_trees, n_samples)
        # We need to find the most common prediction for each sample across trees
        final_predictions = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)
        return final_predictions

    # Optional: Implement predict_proba if needed (average probabilities from trees)
    # This requires DecisionTree to also calculate probabilities at leaves.
    # def predict_proba(self, X):
    #     # ... implementation ...
    #     pass

class NaiveBayes:
    """Multinomial Naive Bayes Classifier implemented from scratch."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._classes = None
        self._log_priors = None
        self._log_likelihoods = None # Shape: (n_classes, n_features)
        self._total_feature_counts_per_class = None # Total count of all features for each class
        self._n_features = None

    def fit(self, X, y):
        # print(f"Fitting Naive Bayes (alpha={self.alpha})...")
        # start_time = time.time()
        y_np = y.values if isinstance(y, pd.Series) else np.array(y)
        is_sparse = hasattr(X, "tocsr")

        n_samples, self._n_features = X.shape
        self._classes = np.unique(y_np)
        n_classes = len(self._classes)

        self._log_priors = np.zeros(n_classes)
        # Stores sum of counts for each feature for each class
        feature_counts_per_class = np.zeros((n_classes, self._n_features))

        for idx, c in enumerate(self._classes):
            if is_sparse:
                X_c = X[y_np == c]
            else:
                X_np = X if isinstance(X, np.ndarray) else np.array(X)
                X_c = X_np[y_np == c]

            n_samples_c = X_c.shape[0]
            self._log_priors[idx] = np.log(n_samples_c / n_samples)

            # Sum feature counts for class c (axis=0)
            # Need to handle sparse and dense differently for sum
            if is_sparse:
                # For sparse matrices (like TF-IDF output), sum along axis 0
                feature_counts_per_class[idx, :] = X_c.sum(axis=0)
            else:
                feature_counts_per_class[idx, :] = X_c.sum(axis=0)

        # Calculate total feature counts for each class (sum across features)
        self._total_feature_counts_per_class = feature_counts_per_class.sum(axis=1)

        # Calculate log likelihoods with Laplace smoothing
        numerator = feature_counts_per_class + self.alpha
        # Denominator: total features in class + alpha * total number of unique features
        denominator = self._total_feature_counts_per_class[:, np.newaxis] + self.alpha * self._n_features
        self._log_likelihoods = np.log(numerator / denominator)
        # end_time = time.time()
        # print(f"Naive Bayes fitting completed in {end_time - start_time:.2f} seconds.")

    def _predict_log_proba(self, X):
        if self._log_priors is None or self._log_likelihoods is None:
            raise Exception("Naive Bayes model must be fitted before prediction.")

        # Check if X is sparse and convert if necessary for matrix multiplication
        if hasattr(X, "tocsr"):
            X_matrix = X
        elif isinstance(X, np.ndarray):
            X_matrix = X # Assume dense
        else:
             X_matrix = np.array(X) # Convert other types

        # Calculate log probabilities for each class
        # log P(c | x) ~ log P(c) + sum(log P(xi | c) * xi_count)
        # Using matrix multiplication for efficiency: X @ log_likelihoods.T
        # Ensure log_likelihoods has shape (n_classes, n_features)
        # Ensure X_matrix has shape (n_samples, n_features)
        log_probs = X_matrix @ self._log_likelihoods.T + self._log_priors
        return log_probs # Shape: (n_samples, n_classes)

    def predict(self, X):
        log_probs = self._predict_log_proba(X)
        # Return the class with the highest log probability
        return self._classes[np.argmax(log_probs, axis=1)]

    def predict_proba(self, X):
        log_probs = self._predict_log_proba(X)
        # Convert log probabilities to probabilities (handle potential underflow/overflow)
        # Subtract max log prob for numerical stability before exp
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - max_log_probs)
        # Normalize to sum to 1
        probs_sum = np.sum(probs, axis=1, keepdims=True)
        return probs / probs_sum


