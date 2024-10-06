import pandas as pd
import numpy as np
import argparse
import pickle
import os
from data_preprocessing import preprocessing

# Since predict.py will parse user-inputted arguments, defining functions which take each command line argument as input

def load_model(model_path):
    with open(model_path, 'rb') as f:
        regression_model = pickle.load(f)
    return regression_model

def classification_results(df, output_path):
    df.to_csv(output_path, index=False)



# Functions to calculate predictions using the coefficients
def Bias_Term(x): ## Offsets entire data by intercept (Same as in train_model.py)
    if (len(x.shape)==1):
        x=x[:,np.newaxis]
    b=np.ones((x.shape[0],1)) # Creating a new column of ones
    x=np.concatenate((b,x), axis=1) # Concatenating column to feature-matrix
    return x

def Predict(X,b):
    return (np.dot(X,b)) # Equation (43) from PDF (multiplying features with coefficients)

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
    coefficients = model_data['coefficients']
    feature_names = model_data['features']

    # Load data from the specified path
    # output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
    output_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # Temporarily saving processed data in same folder
    
    # Process the data as we have for the train set
    processed_data_dir = preprocessing(args.data_path, output_dir)
    data = pd.read_csv(processed_data_dir)
    
    # Splitting the features and output variable (fuel consumption)
    X = data[feature_names].values
    print(X)
    X_bias = Bias_Term(X)
    y = data['FUEL CONSUMPTION'].values

    # Make predictions
    predicted_values = Predict(X_bias, coefficients)

    def metrics(actual, predicted): # Calculating r2, MSE, RMSE from 2 arrays: predicted values and actual.
        tss = np.sum((actual - np.mean(actual))**2)
        ssr = np.sum((predicted - actual)**2)
        r2 = 1 - (ssr / tss)
        mse = np.mean((predicted - actual)**2)
        return mse, np.sqrt(mse), r2

    mse, rmse, r2 = metrics(y, predicted_values)

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Actual': y,
        'Predicted': predicted_values
    })

    regression_results(predictions_df, args.predictions_output_path)

    # Save metrics to the metrics.txt file)
    save_metrics(mse, r2, args.metrics_output_path)

    print(f"Predictions saved to {args.predictions_output_path}")
    print(f"Metrics saved to {args.metrics_output_path}")
