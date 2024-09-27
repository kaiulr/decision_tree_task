import pandas as pd
import numpy as np
import os
from pathlib import Path


output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data') 

df = pd.read_csv("C:/Users/Fiona/Desktop/Fiona_Arora_A1/fraud_train.csv")

df=df.sample(n=800000, random_state=0)

## Checking missing values and dtypes
null_cells = df.isnull().sum()
print(null_cells)
print("No missing values found, so no interpolation/filling in required")

dtypes = df.dtypes
print(dtypes)
# All dtypes in columns appear correct.

# Noticed small typo in 'oldbalanceOrg' column name
df = df.rename(columns={'oldbalanceOrg':'oldbalanceOrig'})

## Deleting highly branched attributes, e.g. the receiver/sender name
print(f"Number of unique values in the account name columns: \nnameOrig: {df.nameOrig.nunique()} \nnameDest: {df.nameDest.nunique()}")
print("Deleting highly branched column to avoid overfitting.")

## The below preprocessing was done after the exploration revealed high correlation within IVs
# To glean other information from the highly correlated variables (oldbalorg, newbalorg and oldbaldest, newbaldest)
# Finding differences (which should be equal to the 'amount' column)

# Origin a/c - New - Old Balance
df['orig_delta'] = df['oldbalanceOrig']-df['newbalanceOrig']

# Destination a/c - New - Old Balance
df['dest_delta'] = df['oldbalanceDest']-df['newbalanceDest']

## Experimenting with a variable that checks if the 'amount' column and account balance changes are similar
# As opposed to conditioning on each type of transaction, checking if either of the deltas (orig and dest) are equal to the amount.
df['amount_is_delta'] = df.amount==np.abs(df.orig_delta)

## Converting categorical data to numeric
# Coding 'type' column into 10 buckets (first experimentation)
# df['coded_type'] = pd.Categorical(df['type'])
df['coded_type'], _ = pd.factorize(df['type'])

## Converting numeric data into categories (bucketing values)
# Splitting the 'amount' and account balance deltas into 5 equal buckets
# Using pd.qcut instead of pd.cut to ensure equal bin depth

df['orig_delta_split'] = pd.qcut(df['orig_delta'], 10, labels=[i for i in range(1,9)], retbins=False, precision=3, duplicates='drop')
# df['dest_delta_split'] = pd.qcut(df['dest_delta'], 10, labels=[i for i in range(1,9)], retbins=False, precision=3, duplicates='drop')

# Check how many buckets are actually created with qcut on amount, then check 
# n = pd.qcut(df['amount'], 10, retbins=False, precision=3, duplicates='drop').nunique()
# print(n)
# df['amount_split'] = pd.qcut(df['amount'], 10, labels=[i for i in range(1,n-1)], retbins=False, precision=3, duplicates='drop')


# Notice that step is referring to hours since some inception point at which transactions were recorded.
# Use this to convert step into hours in a day, beginning with 12-hour periods which indicate, broadly, first half and second half of a 'day'
# Also add a 6-hour, 3-hour, and 1-hour window for further experimentation

def map_to_interval(step, interval_size):
    return (step % 24) // interval_size + 1

# Add new columns for each interval size (1-hour, 2-hour, 3-hour, 6-hour)
df['1_hour_step'] = df['step'].apply(lambda x: map_to_interval(x, 1))
df['3_hour_step'] = df['step'].apply(lambda x: map_to_interval(x, 3))
df['6_hour_step'] = df['step'].apply(lambda x: map_to_interval(x, 6))
df['12_hour_step'] = df['step'].apply(lambda x: map_to_interval(x, 12))


df.to_csv(os.path.join(output_dir, 'training_data.csv'), index=False)