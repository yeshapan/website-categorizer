#to save and load dataset csv

import pandas as pd #for dataframe handling
import os
from pathlib import Path #for managing file paths (esp clean for cross-platform)

DATA_PATH = Path("websites_dataset.csv")

def load_dataset() -> pd.DataFrame: #load dataset as pandas df
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame(columns=["url", "text", "label"]) #for 1st iteration - file doesn't exist so it'll create a df with these 3 columns

#add a new record to df
def add_entry(url: str, text: str, label: str):
    df = load_dataset()
    new_row = pd.DataFrame([[url, text, label]], columns=df.columns) #create new row (use passed values)
    df = pd.concat([df, new_row], ignore_index=True) #combine new row into existing df
    #ignore_index=True reassigns row numbers cleanly
    df.to_csv(DATA_PATH, index=False) #save the updated df back to csv file
    #index=False prevents pandas from writing row nums back into csv (coz not needed)

def get_dataset_size():
    if not os.path.isfile(DATA_PATH):
        return 0

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return sum(1 for line in f) - 1  # minus header
