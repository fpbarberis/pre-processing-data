import pandas as pd
import numpy as np
from path import MAIN_PATH

# read data
dataset_url = rf'\data\Updated_sales.csv'
CHUNK_SIZE = 1000
dataset = pd.read_csv(MAIN_PATH+dataset_url, chunksize=CHUNK_SIZE)

# initialize an empty list to store the processed chunks
processed_chunks = []

# append the processed chunk to the list
for chunk in dataset:
    chunk = chunk.loc[chunk["Order ID"] != "Order ID"].dropna()
    processed_chunks.append(chunk)

# concatenate the processed chunks into a single DataFrame
df = pd.concat(processed_chunks, axis=0)

# make categorical
df_processed = df.copy()
df_processed["State"] = df_processed["Purchase Address"].str.rsplit().str.get(-2)
df_processed["State_Encoded"] = pd.Categorical(df_processed["State"]).codes

# feature engineering for regression analysis
''' Regression analysis is a statistical technique used to model the relationship between one or more input variables 
(also known as independent variables or predictors) and a continuous output variable
(also known as the dependent variable or response). 
In machine learning, regression analysis is used to predict a numerical variable, like the price of a house based on its size, 
number of rooms, and other features. '''

columns_to_process = ["Quantity Ordered", "Price Each"]
for column in columns_to_process:
    df_processed[f"log_{column}"] = np.log(
        df_processed[column].astype("float"))

print(df_processed)

''' The natural logarithm transformation can help in regression analysis and linearize the relationship between 
the input variables and the output variable. Transforming the input variables can improve the performance of the regression model 
by reducing the effects of nonlinearity and making the relationships between the variables more interpretable. '''
