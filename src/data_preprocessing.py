import pandas as pd
import numpy as np
import os
from pathlib import Path


output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data') 

df = pd.read_csv("C:/Users/Fiona/Desktop/Fiona_Arora_A1/fraud_train.csv")

df=df.sample(n=200000, random_state=0)

## Checking missing values and dtypes
null_cells = df.isnull().sum()
print(null_cells)
# No null cells found

dtypes = df.dtypes
print(dtypes)
# All dtypes in columns appear correct.

print(f"% of fraudulent transactions flagged: {np.mean(df['isFlaggedFraud'])}")
# isFlaggedFraud has no information, drop the column

# Noticed small typo in 'oldbalanceOrg' column name
df = df.rename(columns={'oldbalanceOrg':'oldbalanceOrig'})

# Encoding Type column to aid with writing decision tree code
df = pd.get_dummies(df, columns='type', drop_first=True)

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

## The below preprocessing was done after the exploration revealed high correlation within IVs

# Origin a/c - New - Old Balance
df['orig_delta'] = df['oldBalanceOrig']-df['newBalanceOrig']

# Destination a/c - New - Old Balance
df['dest_delta'] = df['oldBalanceDest']-df['newBalanceDest']
