import pandas as pd
import numpy as np
import argparse
import pickle
import os
from data_preprocessing import preprocessing
# from train_model import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Since predict.py will parse user-inputted arguments, defining functions which take each command line argument as input

def load_model(model_path):
    with open(model_path, 'rb') as f:
        regression_model = pickle.load(f)
    return regression_model

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
    
    # Splitting the features and output variable (fuel consumption)
    data = data[columns]
    we=data.to_numpy()
    we=we.astype(np.float64)
    X = we[:, 1:]
    y = we[:, 1]

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
