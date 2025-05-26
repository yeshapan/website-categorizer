# main program for user CLI

from categorizer.scraper import scrape_website
from categorizer.preprocess import preprocess_text
from categorizer.dataset import add_entry, load_dataset
from categorizer.model import train_model

def main():
    print("\n Website Categorizer")
    url = input("Enter website URL: ").strip()
    text = scrape_website(url) #scrape text

    if not text:
        print("Failed to extract website text.")
        return

    cleaned = preprocess_text(text)
    print("\nSample of cleaned text:")
    print(cleaned[:500] + "...\n") #display initial 500 char (for user to see extracted text preview kinda)

    label = input("Manually enter category (e.g. sports, ecommerce, medical...): ").strip().lower() #entered manually during dataset creation
    add_entry(url, cleaned, label)

    df = load_dataset()
    print(f"\nDataset size: {len(df)} records") #load csv and show total num of rows yet (for CLI phase rn)

    if len(df) >= 200:  #for now we'll train on atleast 200 samples for better accuracy
        print("\n Training model...")
        train_model(df)
    else:
        print("Not enough data to train yet (need at least 200).")

if __name__ == "__main__":
    main()
