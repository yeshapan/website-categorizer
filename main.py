# main program for user CLI
import pandas as pd
from categorizer.scraper import scrape_website
from categorizer.preprocess import preprocess_text
from categorizer.dataset import add_entry, get_dataset_size
from categorizer.model import train_model

def get_user_input():
    url= input("\nEnter website URL: ").strip()
    raw_text = scrape_website(url) #scrape text

    if not raw_text:
        print("Failed to extract website text.")
        return

    print("\nSample of cleaned text:")
    text= preprocess_text(raw_text) #cleaned text
    print(text[:500] + "...\n") #display initial 500 char (for user to see extracted text preview kinda)

    category = input("Manually enter category (e.g. ecommerce, medical, technology, sports, etc): ").strip().lower() #entered manually during dataset creation
    return url, text, category

def main():
    print("\n Website Categorizer \n")

    try:
        num_entries= int(input("How many websites would you like to input this time? "))
    except ValueError:
        print("Please enter a valid number.")
        return

    for i in range(num_entries):
        print(f"\nEntry {i+1} of {num_entries}")
        url, text, category = get_user_input()
        add_entry(url, text, category)
    

    size = get_dataset_size()
    print(f"\nDataset size: {size} records")

    if size >= 150:
        choice= input("Do you want to start training the model now? (y/n): ").strip().lower()
        if choice== "y":
            print("Training model now..")
            df=pd.read_csv("websites_dataset.csv")
            train_model(df)
        else:
            print("Okay you can add more data if you want")
    else:
        print("Not enough data to train yet (need at least 150 records).")

    
if __name__ == "__main__":
    main()
