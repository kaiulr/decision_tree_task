import pandas as pd
import numpy as np
import argparse
import pickle
import os
from data_preprocessing import preprocessing
from collections import Counter
from itertools import combinations

# from train_model import DecisionTreeClassifierBinary
# from train_model import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Since predict.py will parse user-inputted arguments, defining functions which take each command line argument as input

def load_model(model_path):
    with open(model_path, 'rb') as f:
        classification_model = pickle.load(f)
    return classification_model

def classification_results(df, output_path):
    df.to_csv(output_path, index=False)

## Defining functions to find all important decision tree metrics
# Accuracy = TP+TN / TP+TN+FN+FP
def accuracy_score(y_true, y_pred):
    if len(y_true)!=0:
        return np.sum(y_true == y_pred) / len(y_true) # Correct predictions by the entire size of the dataset
    else:
        return 'NA (TP+FN=0)'

# Precision = TP / TP+FP
def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives
    if tp + fp == 0:
        return 0 # In case no positives detected in entire dataset
    return tp / (tp + fp)

# Recall = TP / TP+FN
def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives
    if tp + fn == 0:
        return 0 #In case both the entire dataset has no 'positive' values indicated 1
    return tp / (tp + fn)

# F1 Score: Harmonic Mean of the precision and recall
# F1 = 2*(P*R)/P+R
def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def confusion_matrix(y_true, y_pred):
    arr = [[[np.sum((y_true==0) & (y_pred == 0))], [np.sum((y_true==0) & (y_pred == 1))]], 
           [[np.sum((y_true==1) & (y_pred == 0))], [np.sum((y_true==1) & (y_pred == 1))]]]
    return arr

def save_metrics(actual, predicted, output_path):
    prec = precision_score(actual, predicted)
    rec = recall_score(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    conf_matrix = confusion_matrix(actual, predicted)
    classification_metrics = ['Classification Metrics:\n', f'Accuracy: {accuracy}\n', 
                      f'Precision: {prec}\n', f'Recall: {rec}\n', f'F1-Score: {f1}\n', f'Confusion Matrix:\n[{conf_matrix[0]},\n{conf_matrix[1]}]']
    
    with open(output_path, 'w') as f:
        f.writelines(classification_metrics)

def Predict(tree, data):
    return tree.predict(data)


## Defining the node class for the decision tree:
# Creating a binary tree, where the test.conditions (as seen in Slide 34 of Lecture 5 & 6)
# are stored in 'self.categories' (so that if a node contains those categories)
# The node label is stored in self.value
class TreeNode:
    def __init__(self, feature_index=None, categories=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index  # Feature index used for the split
        self.categories = categories        # Categories that go to the left node (binary split)
        self.left = left
        self.right = right
        self.value = value  # If the node is a leaf, this holds the class label

class DecisionTreeClassifierBinary:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def gini_impurity(self, y):
        """Calculate the Gini Impurity for a list of class labels."""
        n_samples = len(y)
        if n_samples == 0:
            return 0
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / n_samples
        return 1.0 - np.sum(probabilities ** 2)
    
    def entropy(self, y):
        """Calculate the entropy for a list of class labels."""
        n_samples = len(y)
        if n_samples == 0:
            return 0
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / n_samples
        return -np.sum(probabilities * np.log(probabilities))

    def split_dataset(self, X, y, feature_index, categories):
        """Split the dataset into two subsets based on the given feature and categories (binary split)."""
        left_indices = np.isin(X[:, feature_index], categories)  # Left node for categories in the group
        right_indices = ~left_indices # Right node all other categories
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

    def best_split(self, X, y):
        """Find the best binary split for categorical features."""
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature_index = None
        best_categories = None
        best_splits = None

        for feature_index in range(n_features):
            # Get all unique categories for the current feature
            categories = np.unique(X[:, feature_index])

            # We need to find the best binary split of categories
            # Generate all possible binary splits (groupings) of categories
            if len(categories) > 1:  # Only need to split if there is more than one category
                for group_size in range(1, len(categories)):
                    for left_categories in combinations(categories, group_size):
                        X_left, X_right, y_left, y_right = self.split_dataset(X, y, feature_index, left_categories)
                        
                        if len(y_left) == 0 or len(y_right) == 0:
                            continue

                        # Calculate the weighted Gini Impurity
                        gini_left = self.gini_impurity(y_left)
                        gini_right = self.gini_impurity(y_right)
                        weighted_gini = (len(y_left) / n_samples) * gini_left + (len(y_right) / n_samples) * gini_right

                        if weighted_gini < best_gini:
                            best_gini = weighted_gini
                            best_feature_index = feature_index
                            best_categories = left_categories
                            best_splits = (X_left, X_right, y_left, y_right)

        return best_feature_index, best_categories, best_splits

    def fit(self, X, y, depth=0):
        # This function builds the tree recursively, similar to the algorithm in the slides
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions: If all the instances in the node are of one label
        # Or if all the attributes/features in the data have been exhausted
        # Or if the self-specified 'maximum depth' of the tree has been reached
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self.most_common_label(y)
            return TreeNode(value=leaf_value)

        # Finding the best split
        feature_index, categories, splits = self.best_split(X, y)
        if splits is None:
            leaf_value = self.most_common_label(y)
            return TreeNode(value=leaf_value)

        X_left, X_right, y_left, y_right = splits

        # Create subtrees recursively
        left_subtree = self.fit(X_left, y_left, depth + 1)
        right_subtree = self.fit(X_right, y_right, depth + 1)

        return TreeNode(feature_index, categories, left_subtree, right_subtree)

    def predict(self, X):
        # Recursively finding the class label for instances (sample of the entire dataset) in x
        return np.array([self._traverse_tree(sample, self.root) for sample in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        feature_value = x[node.feature_index]
        if feature_value in node.categories:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def most_common_label(self, y):
        # Find most common label in a sample for a given node
        counts = Counter(y)
        return counts.most_common(1)[0][0]

if __name__ == "__main__":
    
    # This function parses the command line arguments, in the exact format specified.

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data (CSV file)')
    parser.add_argument('--metrics_output_path', type=str, required=True, help='Path to save metrics')
    parser.add_argument('--predictions_output_path', type=str, required=True, help='Path to save predictions')

    args = parser.parse_args()

    # Load saved model from path
    model_data = load_model(args.model_path)
    tree  = model_data['tree']
    columns = model_data['columns']

    # Load data from the specified path
    # output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    output_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # Temporarily saving processed data in same folder
    
    # Process the data as we have for the train set
    processed_data_dir = preprocessing(args.data_path, output_dir)
    data = pd.read_csv(processed_data_dir)
    
    # Splitting the features and output variable (isFraud)
    data = data[columns]
    we=data.to_numpy()
    we=we.astype(np.float64)
    X = we[:, 1:]
    y = we[:, 0]

    # Make predictions
    predicted_values = Predict(tree, X)

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Actual': y,
        'Predicted': predicted_values
    })

    classification_results(predictions_df, args.predictions_output_path)

    # Save metrics to the metrics.txt file)
    save_metrics(y, predicted_values, args.metrics_output_path)

    print(f"Predictions saved to {args.predictions_output_path}")
    print(f"Metrics saved to {args.metrics_output_path}")
