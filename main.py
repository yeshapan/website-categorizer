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

    while True:
        size = get_dataset_size()
        print(f"\nCurrent dataset size: {size} records")

        if size >= 150:
            choice = input("Do you want to start training the model now? (y/n): ").strip().lower()
            if choice == 'y':
                print("Training model now..")
                df = pd.read_csv("websites_dataset.csv")
                train_model(df)
                break  #exit after training
            elif choice == 'n':
                print("Okay, let's add more data then.")
                #continue to ask how many websites below
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
                continue  #re-ask this question

        # if size<150 or user chose 'n'
        try:
            num_entries = int(input("How many websites would you like to input this time? "))
        except ValueError:
            print("Please enter a valid number.")
            continue  # re-ask

        for i in range(num_entries):
            print(f"\nEntry {i + 1} of {num_entries}")
            user_input = get_user_input()
            if user_input:
                url, text, category = user_input
                add_entry(url, text, category)
            else:
                print("Skipping entry due to extraction failure or missing data.")



    
if __name__ == "__main__":
    main()
