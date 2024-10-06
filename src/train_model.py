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
data_dir = preprocessing(input_file, output_dir)
df = pd.read_csv(data_dir)
print(df)
## Defining the node class for the decision tree:
# Creating a binary tree, where the test.conditions (as seen in Slide 34 of Lecture 5 & 6)
# are stored in 'self.categories' (so that if a node contains those categories)
# The node label is stored in self.label
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

    def split_dataset(self, X, y, feature_index, categories):
        """Split the dataset into two subsets based on the given feature and categories (binary split)."""
        left_indices = np.isin(X[:, feature_index], categories)  # Left node for categories in the group
        right_indices = ~left_indices                            # Right node for other categories
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


# input_dir = os.path.join(os.path.dirname((os.path.abspath("")))) 
# df = pd.read_csv(os.path.join(input_dir, 'data', 'training_data.csv'))

df = df[['isFraud', 'coded_type', 'orig_delta_split', 'amount_is_delta','3_hour_step']]

we=df.to_numpy()
we=we.astype(np.float64)
X = we[:, 1:]
y = we[:, 1]

trainx = we[:599999, 1:] # Input matrix of all features
trainy = we[:599999, 0] # Output vector of fuel transmission
testx = we[600000:, 1:]
testy = we[600000:, 0]

# decision_tree = tree.DecisionTreeClassifier()
# decision_tree.fit(xtrain, ytrain)

# Y_pred_dt = decision_tree.predict(xtest)
# decision_tree_score = decision_tree.score(xtest, ytest) * 100
# print(decision_tree_score)

# Train the classifier
# clf.fit(xtrain, ytrain)

# # Make predictions
# y_pred = clf.predict(xtest)

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

save_results(ytest, predictions)

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
    folds = create_folds(X)
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
    
cross_validation(X, y, 10)



model1_data = {
    'tree':clf
}

with open(os.path.join(model_dir, 'decision_tree_model1.pkl'), 'wb') as f:
    pickle.dump(model1_data, f)

print("Model saved to 'decision_tree_model1.pkl'")
