import pandas as pd
import numpy as np
import os
from pathlib import Path


# output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data') 

# df = pd.read_csv("C:/Users/Fiona/Desktop/Fiona_Arora_A1/decision_tree_task/fraud_train.csv")

def preprocessing(input_dir, output_dir):
    df = pd.read_csv(input_dir)
    # df=df.sample(n=800000, random_state=0)

    ## Checking missing values and dtypes
    null_cells = df.isnull().sum()
    print("Number of Null Values in Each Column:")
    print(null_cells)
    
    # No null cells found in the data set provided. If the test has null values, we can interpolate by running this command:
    if null_cells.sum() != 0:
        df.fillna(df.mean(), inplace=True)

    dtypes = df.dtypes
    print("Column Datatypes:")
    print(dtypes)
    # All dtypes in columns appear correct.

    # Noticed small typo in 'oldbalanceOrg' column name
    df = df.rename(columns={'oldbalanceOrg':'oldbalanceOrig'})

    ## Deleting highly branched attributes, e.g. the receiver/sender name
    print(f"Number of unique values in the account name columns: \nnameOrig: {df.nameOrig.nunique()} \nnameDest: {df.nameDest.nunique()}")
    # Unique values in nameDest column<rows in DF.
    # To check if this information might be useful for prediction, creating columns for indicating '1' if nameOrig/nameDest is duplicate or not
    df['nameOrig_duplicate'] = df['nameOrig'].duplicated(keep=False)
    df['nameDest_duplicate'] = df['nameDest'].duplicated(keep=False)

    # Selecting features:
    # print("Deleting highly branched column to avoid overfitting.")
    df = df.drop(columns=['nameDest','nameOrig','isFlaggedFraud'])

    # Creating feature on whether account value was depleted to 0
    df['fully_depleted'] = np.where(df['amount'] == df['oldbalanceOrig'], True, False)

    ## The below preprocessing was done after the exploration revealed high correlation within IVs
    # To glean other information from the highly correlated variables (oldbalorg, newbalorg and oldbaldest, newbaldest)
    # Finding differences (which should be equal to the 'amount' column)

    # Origin a/c - New - Old Balance
    df['orig_delta'] = df['oldbalanceOrig']-df['newbalanceOrig']

    # Destination a/c - New - Old Balance
    df['dest_delta'] = df['oldbalanceDest']-df['newbalanceDest']

    # Creating feature for if amount transacted was 10,000,000 and the transaction type was 'transfer'
    # And if destination balance remained unchanged
    df['transaction_10mn'] = np.where(((df['amount']==10000000) & (df['type']=='TRANSFER') & (df['dest_delta']==0)), True, False)
 
    ## Experimenting with a variable that checks if the 'amount' column and account balance changes are similar
    # As opposed to conditioning on each type of transaction, checking if either of the deltas (orig and dest) are equal to the amount.
    df['amount_is_delta'] = df.amount==np.abs(df.orig_delta)

    ## Converting categorical data to numeric
    # Coding 'type' column into 10 buckets (first experimentation)
    df['coded_type'], _ = pd.factorize(df['type'])

    ## Converting numeric data into categories (bucketing values)
    # Splitting the 'amount' and account balance deltas into 5 equal buckets
    # Using pd.qcut instead of pd.cut to ensure equal bin depth
    # Dropping duplicates since bins need to be entirely unique for decision trees to be able to parse through the bins

    # Orig_Delta
    min_value = df['orig_delta'].min()
    max_value = df['orig_delta'].max()

    number_of_bins = 10
    bins = np.linspace(min_value, max_value, number_of_bins + 1)
    df['orig_delta_split'] = pd.cut(df['orig_delta'], bins=bins, include_lowest=True)

    # Dest_Delta
    min_value = df['dest_delta'].min()
    max_value = df['dest_delta'].max()

    number_of_bins = 10
    bins = np.linspace(min_value, max_value, number_of_bins + 1)
    df['dest_delta_split'] = pd.cut(df['dest_delta'], bins=bins, include_lowest=True)

    # Amount
    min_value = df['dest_delta'].min()
    max_value = df['dest_delta'].max()

    number_of_bins = 10
    bins = np.linspace(min_value, max_value, number_of_bins + 1)
    df['dest_delta_split'] = pd.cut(df['dest_delta'], bins=bins, include_lowest=True)

    ## Transforming the 'step' variable

    # Since 'step' is referring to hours since some inception point at which transactions were recorded.
    # Use this to convert step into hours in a day, beginning with 12-hour periods which indicate, broadly, first half and second half of a 'day'
    # Also add a 6-hour, 3-hour, and 1-hour window for further experimentation

    def map_to_interval(step, interval_size):
        return (step % 24) // interval_size + 1

    # Add new columns for each interval size (1-hour, 2-hour, 3-hour, 6-hour)
    df['1_hour_step'] = df['step'].apply(lambda x: map_to_interval(x, 1))
    df['3_hour_step'] = df['step'].apply(lambda x: map_to_interval(x, 3))
    df['6_hour_step'] = df['step'].apply(lambda x: map_to_interval(x, 6))
    df['12_hour_step'] = df['step'].apply(lambda x: map_to_interval(x, 12))

    # 1-hour step binary
    df['1_hour_step_binary'] = np.where(df['1_hour_step']<=10, True, False)

    # Type binary
    df['type_binary'] = np.where(df['type'].isin(['CASH_OUT','TRANSFER']), True, False)


    df.to_csv(os.path.join(output_dir, 'training_data.csv'), index=False)
    print(f"Processed data saved to '{os.path.join(output_dir, 'training_data.csv')}'")

    return os.path.join(output_dir, 'training_data.csv')