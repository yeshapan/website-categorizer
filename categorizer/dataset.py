#to save and load dataset csv

import pandas as pd
import os
from pathlib import Path

DATA_PATH = Path("websites_dataset.csv")

def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        #FIX: Changed to 'utf-8-sig' to handle scraped text AND Excel compatibility.
        #Do not use 'latin1' here, or it will corrupt UTF-8 data on load.
        try:
            df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
        except UnicodeDecodeError:
            # Fallback in case the file is currently saved in a legacy format
            df = pd.read_csv(DATA_PATH, encoding='latin1')

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
    #FIX: Use 'utf-8-sig' here as well. Consistency is key.
    #This ensures Excel opens it correctly without showing weird symbols.
    df.to_csv(DATA_PATH, index=False, encoding='utf-8-sig')

def get_dataset_size() -> int:
    if not DATA_PATH.is_file(): #use is_file() for Path objects
        return 0
    try:
        #a more robust way to count lines in a CSV, pandas can also do this
        #FIX: Match the encoding used in load_dataset
        df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
        return len(df)
    except pd.errors.EmptyDataError:
        return 0
    except Exception: #catch other potential read errors
        #fallback for potentially malformed CSV, though less likely
        try:
            with open(DATA_PATH, "r", encoding="utf-8-sig") as f:
                return sum(1 for _ in f) - 1 #minus header
        except:
             return 0