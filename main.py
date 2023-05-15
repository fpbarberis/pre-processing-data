import pandas as pd
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
