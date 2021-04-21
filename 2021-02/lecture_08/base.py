import tqdm
import random
from scipy import stats
import numpy as np


def f_entropy(values):
    """
    Calculate information entropy for list of target values (entropy = - sum_i(p_i * log(p_i)))
    :param values: array-like target values, e.g. [0, 1, 0, 1, 1, 0]
    :return float value of entropy
    """
    # Find each class probability
#     p = np.bincount(p) / float(p.shape[0])
    p = np.bincount(values) / len(values)

    ep = stats.entropy(p)
    if ep == -float('inf'):
        return 0.0
    return ep


def information_gain(y, splits):
    """
    Calculate the information gain for a split of 'y' to 'splits' (gain = f(y) - (n_1/n)*f(split_1) - (n_2/n)*f(split_2) ...)
    :param y: array-like initial target values, e.g. [0, 1, 0, 1, 1, 0]
    :param splits: array-like list of splits, e.g. [[0, 1, 0], [1, 1, 0]]
    :return float value of information gain
    """
#     splits_entropy = sum([f_entropy(split) * (float(split.shape[0]) / y.shape[0]) for split in splits])
    splits_entropy = sum([f_entropy(split) * len(split) for split in splits]) / len(y)
    return f_entropy(y) - splits_entropy    


def split(X, y, threshold):
    """
    Make a binary split of (X, y) using the threshold
    :param X: array-like feature values for all objects (1 feature)
    :param y: array-like target values for all objects (e.g. [0, 1, 0, 1, 1, 0])
    :param threshold: float threshold for splitting 
    :return array-like target values (y) of each split (e.g. [[0, 1, 0], [1, 1, 1]])
    """
    left_mask = (X < threshold)
    right_mask = (X >= threshold)
    return y[left_mask], y[right_mask]


def split_dataset(X, target, column, value):
    """
    Split the dataset (X, target) using X[column] feature at threshold=value
    :param X: array-like features of objects
    :param target: dict {'y': array-like of targets}
    :param column: int index of feature in X
    :param value: float value of threshold for X[column]
    :return array-like features and targets of left and right splits 
    """
    left_mask, right_mask = get_split_mask(X, column, value)

    left, right = {}, {}
    for key in target.keys():
        left[key] = target[key][left_mask]
        right[key] = target[key][right_mask]
    left_X, right_X = X[left_mask], X[right_mask]
    return left_X, right_X, left, right
    
    
def get_split_mask(X, column, value):
    left_mask = (X[:, column] < value)
    right_mask = (X[:, column] >= value)
    return left_mask, right_mask


class DecisionTree(object):
    """Recursive implementation of decision tree."""

    def __init__(self, regression=False, criterion_name='entropy'):
        """
        :param regression: predict regression values instead of classification (default=False) (TODO: add regression=True support)
        :param criterion_name: criterion to use for splitting (default='entropy') (TODO: add criterion_name='gini' support)
        """
        self.regression = regression
        self.gain = 0
        self.size = 0
        self.column_index = None
        self.threshold = None
        self.outcome = None
        self.outcome_proba = None
        
        self.left_child = None
        self.right_child = None
        
        self.criterion_name = criterion_name
        self.unique_targets = None
        
    def calc_gain(self, y, splits):
        """
        Calculate criterion gain for splits of y
        :param y: array-like initial target values, e.g. [0, 1, 0, 1, 1, 0]
        :param splits: array-like list of splits, e.g. [[0, 1, 0], [1, 1, 0]]
        :return float value of criterion gain
        """
        if self.criterion_name == 'entropy':
            return information_gain(y, splits)
        else:
            raise NotImplementedError

    @property
    def is_terminal(self):
        """
        Return True if self is leaf else False
        """
        return not bool(self.left_child and self.right_child)

    def _find_splits(self, X):
        """
        Find all possible split values of X 
        :param X: array-like feature vector (X - single feature column)
        :return list of splitting values
        """
        split_values = set()

        # Get unique values in a sorted order
        x_unique = list(np.unique(X))
        for i in range(1, len(x_unique)):
            # Find a point between two values
            average = (x_unique[i - 1] + x_unique[i]) / 2.0
            split_values.add(average)
        return list(split_values)

    def _find_best_split(self, X, target, max_features=None):
        """
        Find best feature and value for a split (greedy algorithm)
        :param X: array-like features (num_obj x num_features)
        :param target: dict {'y': array-like of targets}
        :param max_features: number of features to use for best split search (we'll need it for RandomForest in future)
        :return int feature index, float feature threshold value, float criterion gain (for best split)
        """
        # Sample random subset of features
        if max_features is None:
            max_features = X.shape[1]
        #TODO: proper subset sampling
        subset = np.array(range(max_features))
        
        max_gain, max_col, max_val = None, None, None
        for column in subset:
            split_values = self._find_splits(X[:, column])
            for value in split_values:
                splits = split(X[:, column], target['y'], value)
                gain = self.calc_gain(target['y'], splits)

                if (max_gain is None) or (gain > max_gain):
                    max_col, max_val, max_gain = column, value, gain
        return max_col, max_val, max_gain

    def train(self, X, y, unique_targets=None, max_features=None, min_samples_split=2, max_depth=1000, min_gain=0.0001, verbose=False):
        """
        Build a decision tree from training set
        :param X: array-like dataset (num_obj x num_features)
        :param y: dictionary or array-like target values
        :param unique_targets: array-like unique target values
        :param max_features: int or None, the number of features to consider when looking for the best split
        :param min_samples_split: int, the minimum number of samples required to split an internal node
        :param max_depth: int, maximum depth of the tree
        :param min_gain: float, minimum gain required for splitting
        """

        if not isinstance(y, dict):
            y = {'y': y}
        if unique_targets is None:
            unique_targets = sorted(np.unique(y['y']))
            
        self.size = X.shape[0]
        if max_features is None:
            max_features = X.shape[1]
            
        try: # Exit from recursion using assert syntax
            assert (X.shape[0] >= min_samples_split)
            assert (max_depth > 0)

            column, value, gain = self._find_best_split(X, y, max_features)
            self.gain = gain
            
            if verbose:
                print('inputs: {}, y: {}, max_depth: {}, feature: {}, value: {:.2f}, gain: {:.2f}'
                      .format(self.size, y, max_depth, column, value, gain))

            assert gain is not None
            if self.regression:
                assert (gain != 0)
            else:
                assert (gain > min_gain)

            self.column_index = column
            self.threshold = value

            # Split dataset
            left_X, right_X, left_y, right_y = split_dataset(X, y, column, value)

            # Grow left and right child
            self.left_child = DecisionTree(self.regression, self.criterion_name)
            self.left_child.train(left_X, left_y, unique_targets, max_features, min_samples_split, max_depth - 1, min_gain, verbose)

            self.right_child = DecisionTree(self.regression, self.criterion_name)
            self.right_child.train(right_X, right_y, unique_targets, max_features, min_samples_split, max_depth - 1, min_gain, verbose)
            
        except AssertionError:
            self._calculate_leaf_value(y, unique_targets)
            self.left_child = None
            self.right_child = None

    def _calculate_leaf_value(self, target, unique_targets):
        """
        Find output value for leaf and store it in self.outcome & self.outcome_proba
        :param target: dict {'y': array-like of targets}
        :param unique_targets: array-like unique target values
        """
        if self.regression:
            # Mean value for regression task
            self.outcome = np.mean(target['y'])
        else:
            # Most probable class for classification task
            uniques, counts = np.unique(target['y'], return_counts=True)
            self.outcome = uniques.astype(np.int)[np.argmax(counts)]
            
            # Outcome probabilities
            counts_normed = counts / counts.sum()
            probs = {label: prob for label, prob in zip(uniques, counts_normed)}
            
            self.outcome_proba = []
            for unique_target in unique_targets:
                p = probs[unique_target] if unique_target in probs else 0
                self.outcome_proba.append(p)
            self.outcome_proba = np.asarray(self.outcome_proba)

    def predict_row(self, row):
        """
        Predict for single row
        """
        if not self.is_terminal:
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_row(row)
            else:
                return self.right_child.predict_row(row)
        return self.outcome
    
    def predict_proba_row(self, row):
        """
        Predict probability for single row 
        """
        if not self.is_terminal:
            if row[self.column_index] < self.threshold:
                return self.left_child.predict_proba_row(row)
            else:
                return self.right_child.predict_proba_row(row)
        return self.outcome_proba

    def predict(self, X):
        """
        Predict for X
        """
        result = np.zeros(shape=(X.shape[0]), dtype=np.int)
        for i in range(X.shape[0]):
            result[i] = self.predict_row(X[i, :])
        return result
    
    def predict_proba(self, X):
        """
        Predict probabilities for X
        """
        result = np.zeros(shape=(X.shape[0], 2))
        for i in range(X.shape[0]):
            result[i] = self.predict_proba_row(X[i, :])
        return result