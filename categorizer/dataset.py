#to save and load dataset csv

import pandas as pd
import os
from pathlib import Path

DATA_PATH = Path("websites_dataset.csv")

def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        #ensure column names are consistent (coz our model.py uses 'text' and 'category')
        df = df.rename(columns={"url": "url", "text": "text", "label": "category"}) 
        return df
    else:
        #ensure columns match what add_entry and model expects
        return pd.DataFrame(columns=["url", "text", "category"])

def add_entry(url: str, text: str, category: str):
    df = load_dataset()
    new_row_data = {"url": url, "text": text, "category": category}
    #handle case where df might have other columns not being added now
    new_row = pd.DataFrame([new_row_data], columns=df.columns if not df.empty else ["url", "text", "category"])
    
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

def get_dataset_size() -> int:
    if not DATA_PATH.is_file(): #use is_file() for Path objects
        return 0
    try:
        #a more robust way to count lines in a CSV, pandas can also do this
        df = pd.read_csv(DATA_PATH)
        return len(df)
    except pd.errors.EmptyDataError:
        return 0
    except Exception: #catch other potential read errors
        #fallback for potentially malformed CSV, though less likely
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1 #minus header