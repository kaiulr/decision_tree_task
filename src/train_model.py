import numpy as np
import pandas as pd 
import os 
from itertools import combinations
from data_preprocessing import preprocessing
import pickle
from collections import Counter

input_file = "C:/Users/Fiona/Desktop/Fiona_Arora_A1/decision_tree_task/fraud_train.csv"
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'models')
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
# data_dir = preprocessing(input_file, output_dir)
df = pd.read_csv(os.path.join(output_dir, "training_data.csv"))
# df = pd.read_csv(data_dir)


## Defining the node class for the decision tree:
# Creating a binary tree, where the test.conditions (as seen in Slide 34 of Lecture 5 & 6)
# are stored in 'self.categories' (so that if a node contains those categories)
# The node label is stored in self.value
class TreeNode:
    def __init__(self, feature_index=None, categories=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index  # Feature index used for the split
        self.categories = categories # Categories that go to the left node (binary split)
        self.left = left
        self.right = right
        self.value = value  # If the node is a leaf, this holds the class label

class DecisionTreeClassifierBinary:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease # To reduce overfitting
        self.max_features = max_features
        self.root = None

    def gini_impurity(self, y): # Calculates gini impurity based on formula seen in class
        n_samples = len(y)
        if n_samples == 0:
            return 0
        unique_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / n_samples
        return 1.0 - np.sum(probabilities ** 2)

    def information_gain_gini(self, y, y_left, y_right): # Using formula from slides
        # Calculate the Gini impurity of the current node (parent)
        parent_gini = self.gini_impurity(y)

        # Calculate the weighted average Gini impurity of the child nodes
        n_samples = len(y)
        n_left = len(y_left)
        n_right = len(y_right)

        if n_samples == 0:
            return 0

        weighted_gini = (n_left / n_samples) * self.gini_impurity(y_left) + (n_right / n_samples) * self.gini_impurity(y_right)

        # Information gain is the difference between the parent Gini impurity and the weighted average child Gini impurity
        return parent_gini - weighted_gini

    def split_dataset(self, X, y, feature_index, categories): # Splitting instances (binary)
        left_indices = np.isin(X[:, feature_index], categories)  # Left node for categories in the group
        right_indices = ~left_indices  # Right node all other categories
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

    def best_split(self, X, y): # Finding the best split, recursively, using information gain (as opposed to Gini)
        n_samples, n_features = X.shape
        best_gain = -float('inf')
        best_feature_index = None
        best_categories = None
        best_splits = None

        # Updated code: allow for non-deterministic tree by randomly selecting features from the provided dataset.
        # Two features ('fully_depleted' and 'transaction_10mn' are very strongly correlated with the target in our case, so this is optional.
        
        if self.max_features:
            features = np.random.choice(range(n_features), self.max_features, replace=False)
        else:
            features = range(n_features)

        for feature_index in features:
            # Get all unique categories for the current feature
            categories = np.unique(X[:, feature_index])

            # Finding the best binary split based on information gain
            # Generate all possible binary splits (groupings) of categories using combinations library
            if len(categories) > 1:  # Only need to split if there is more than one category
                for group_size in range(1, len(categories)):
                    for left_categories in combinations(categories, group_size):
                        X_left, X_right, y_left, y_right = self.split_dataset(X, y, feature_index, left_categories)

                        # Overfitting - ensure minimum samples per leaf requirement
                        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                            continue

                        # Calculating the information gain using Gini impurity for the split
                        gain = self.information_gain_gini(y, y_left, y_right)

                        # Update the best split if the current gain is higher
                        if gain > best_gain and gain >= self.min_impurity_decrease:
                            best_gain = gain
                            best_feature_index = feature_index
                            best_categories = left_categories
                            best_splits = (X_left, X_right, y_left, y_right)

        return best_feature_index, best_categories, best_splits

    def fit(self, X, y, depth=0):
        # This function builds the tree recursively, similar to the algorithm in the slides
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions
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
        # Find most common label in a sample for a given node (counts based - whichever label is most prevalent)
        counts = Counter(y)
        return counts.most_common(1)[0][0]


keepcolumns = ['isFraud','coded_type','transaction_10mn', 'fully_depleted','orig_delta_split','amount_is_delta', '3_hour_step']
df = df[keepcolumns]

we=df.to_numpy()
we=we.astype(np.float64)
X = we[:, 1:]
y = we[:, 0]

split_index = int(np.ceil(len(X)*0.8))

trainx = we[:split_index, 1:] # Input matrix of all features
trainy = we[:split_index, 0] # Output vector of 'isFraud'
testx = we[split_index+1:, 1:]
testy = we[split_index+1:, 0]

clf = DecisionTreeClassifierBinary(max_depth=5)
clf.root = clf.fit(trainx, trainy)

predictions = clf.predict(testx)

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

print(precision_score(testy,predictions))
print(recall_score(testy,predictions))
print(accuracy_score(testy,predictions))
print(f1_score(testy,predictions))

def save_results(actual, predicted):
    prec = precision_score(actual, predicted)
    rec = recall_score(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    conf_matrix = confusion_matrix(actual, predicted)
    classification_metrics = ['Classification Metrics:\n', f'Accuracy: {accuracy}\n', 
                      f'Precision: {prec}\n', f'Recall: {rec}\n', f'F1-Score: {f1}\n', f'Confusion Matrix:\n[{conf_matrix[0]},\n{conf_matrix[1]}]']
    
    with open(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'results', 'metrics.txt')), 'w') as f:
        f.writelines(classification_metrics)

save_results(testy, predictions)

## Cross Validation for a standard decision tree, using same fold-creation function as in the regression task

def create_folds(X, n):
    # Gets us an evenly spaced index for the entire length of the input matrix X
    # Equal to the number of data points.
    indices = np.arange(len(X))

    np.random.shuffle(indices) # Using in-built numpy library to shuffle the row indices, in order to create randomly ordered folds

    interval_size = len(X) // n # How many data points we want in each fold

    # Create folds using shuffled indices
    # List of lists, where each sub-list is fold_size long
    # Slicing 'indices' into 1st fold, then jumping ahead 1*fold_size points to get next fold, etc.
    n_folds = [indices[i * interval_size:(i + 1) * interval_size] for i in range(n)]
    
    # Add left over datapoints to the last fold (n_folds[-1])
    # Remaining points are all those that are present after the last n*interval_size index.
    if len(X) % n != 0:
        n_folds[-1] = np.concatenate([n_folds[-1], indices[n * interval_size:]])

    return n_folds


def cross_validation(X, y, n_folds):
    folds = create_folds(X, n_folds)
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(n_folds):
        print(f"Using fold {i} as Test Set.")
        # Iteratively using ith fold as the test set, the rest as the training set
        test_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(n_folds) if j != i]) # combining all remaining folds into one list
    
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        print(f"Train Set Size: {len(train_indices)}, Test Set Size: {len(test_indices)}")

        clf = DecisionTreeClassifierBinary(max_depth=5)
        clf.root = clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        # Get metrics for test
        accuracy_list.append(accuracy_score(y_test, predictions))
        precision_list.append(precision_score(y_test, predictions))
        recall_list.append(recall_score(y_test, predictions))
        f1_list.append(f1_score(y_test, predictions))

    print("Mean Accuracy: ", format(np.mean(accuracy_list), ".2f"))
    print("Mean Precision: ", format(np.mean(precision_list), ".2f"))
    print("Mean Recall: ", format(np.mean(recall_list), ".2f"))
    print("Mean F1 Score: ", format(np.mean(f1_list), ".2f"))

    return accuracy_list, precision_list, recall_list, f1_list
    
# a, b, c, d = cross_validation(X, y, 10)

final_data = {
    'columns':keepcolumns,
    'tree': clf
}

with open(os.path.join(model_dir, 'decision_tree_model_final.pkl'), 'wb') as f:
    pickle.dump(final_data, f)

print("Model saved to 'decision_tree_model_final.pkl'")
