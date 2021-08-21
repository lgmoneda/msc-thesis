import pandas as pd
import numpy as np

from sklearn import metrics
from models.functions import std_agg

class DecisionTree():
    """
    The class that implements the decision tree.

    max_depth: the maximum depth the algorithm can keep performing splits
    min_leaf: the minimum number of examples required in a leaf
    """
    def __init__(self, X, y, row_indexes = [], max_depth=5, min_leaf=5, depth=None, verbose=False, time_column=None):
        if len(row_indexes) == 0: row_indexes = np.arange(len(y))
        if depth == None: depth = 0

        self.X, self.y, self.row_indexes, self.max_depth = X, y, row_indexes, max_depth,
        self.min_leaf, self.depth = min_leaf, depth
        self.n_examples = len(row_indexes)
        self.variables = X.columns
        self.value = np.mean(y[row_indexes])
        self.score = float("inf")
        self.verbose = verbose
        self.time_column = time_column

        if verbose and time_column:
            print("Depth: {}".format(self.depth))
            print("Max Depth: {}".format(self.max_depth))
            print("Node periods distribution")
            print(self.X.loc[self.row_indexes, self.time_column].value_counts().sort_index())

        if self.depth <= self.max_depth:
            self.create_split()

    def create_split(self):
        for idx, variable in enumerate(self.variables): self.find_better_split(variable, idx)
        if self.score == float("inf"): return False
        x = self.split_column

        left_split = np.nonzero(x<=self.split_example)
        right_split = np.nonzero(x>self.split_example)
        self.left_split = DecisionTree(self.X, self.y, self.row_indexes[left_split],
                                       depth=self.depth + 1,
                                       max_depth=self.max_depth,
                                       min_leaf=self.min_leaf,
                                       verbose=self.verbose,
                                       time_column=self.time_column)
        self.right_split = DecisionTree(self.X, self.y, self.row_indexes[right_split],
                                        depth=self.depth + 1,
                                        max_depth=self.max_depth,
                                        min_leaf=self.min_leaf,
                                        verbose=self.verbose,
                                        time_column=self.time_column)


    def find_better_split(self, variable, variable_idx):
        x, y = self.X.loc[self.row_indexes, variable], self.y[self.row_indexes]

        x = x.values
        sorted_indexes = np.argsort(x)
        sorted_x, sorted_y = x[sorted_indexes], y[sorted_indexes]
        right_count, right_sum, right_squared_sum = len(sorted_y), sorted_y.sum(), (sorted_y ** 2).sum()
        left_count, left_sum, left_squared_sum = 0, 0.0, 0.0

        for example in range(0, self.n_examples - self.min_leaf - 1):
            x_i, y_i = sorted_x[example], sorted_y[example]
            left_count += 1
            right_count -= 1
            left_sum += y_i
            right_sum -= y_i
            left_squared_sum += y_i ** 2
            right_squared_sum -= y_i ** 2

            if example < self.min_leaf or x_i == sorted_x[example + 1]:
                continue

            left_std = std_agg(left_count, left_sum, left_squared_sum)
            right_std = std_agg(right_count, right_sum, right_squared_sum)
            current_score = left_std * left_count + right_std * right_count

            if current_score < self.score:
                self.split_variable, self.score, self.split_example = variable, current_score, x_i
                self.split_variable_idx = variable_idx


    @property
    def is_leaf(self): return self.score == float("inf")

    @property
    def split_column(self):
        return self.X.values[self.row_indexes, self.split_variable_idx]

    def predict(self, X):
        if type(X) == np.ndarray:
            X = pd.DataFrame(X, columns=self.variables)
        return np.array([self.predict_row(x) for x in X.iterrows()])

    def predict_row(self, x):
        if self.is_leaf: return self.value
        tree = self.left_split if x[1][self.split_variable] <= self.split_example else self.right_split

        return tree.predict_row(x)

    @property
    def get_split_variable(self):
        if not self.is_leaf:
            return self.split_variable + "@" + self.left_split.get_split_variable + "@" + self.right_split.get_split_variable
        return "LEAF"

    @property
    def feature_importance(self):
        splits = self.get_split_variable
        splits_features = splits.replace("@LEAF", "").split("@")

        return pd.DataFrame(splits_features, columns=["Feature Importance"])["Feature Importance"].value_counts().sort_values(ascending=False)
