# main program for user CLI

import pandas as pd
from categorizer.scraper import scrape_website
from categorizer.preprocess import preprocess_text
from categorizer.dataset import add_entry, get_dataset_size, load_dataset
from categorizer.model import train_model, predict_category_ensemble, clear_prediction_cache, load_trained_models_and_vectorizer

def get_user_input_for_dataset():
    url = input("\nEnter website URL for dataset: ").strip()
    raw_text = scrape_website(url)

    if not raw_text:
        print("Failed to extract website text.")
        return None

    print("\nSample of website text (first 500 chars):")
    print(raw_text[:500] + "...\n")
    
    #preprocessing is done before saving to dataset now, to match model training expectation
    text = preprocess_text(raw_text)
    print("\nSample of cleaned text (first 500 chars):")
    print(text[:500] + "...\n")


    category = input("Manually enter category (e.g. ecommerce, medical, news): ").strip().lower()
    return url, text, category #save preprocessed text

def main_cli():
    print("\nWebsite Categorizer CLI")

    while True:
        print("\nOptions:")
        print("1. Add websites to dataset")
        print("2. Train models")
        print("3. Predict category for a URL")
        print("4. View dataset size")
        print("5. Clear prediction cache")
        print("6. Exit")
        
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            try:
                num_entries = int(input("How many websites would you like to add to the dataset? "))
            except ValueError:
                print("Invalid number. Please enter an integer.")
                continue

            for i in range(num_entries):
                print(f"\nAdding Entry {i + 1} of {num_entries} : ")
                user_input = get_user_input_for_dataset()
                if user_input:
                    url, text, category = user_input
                    add_entry(url, text, category) #text is already preprocessed
                    print(f"Entry for {url} as '{category}' added to dataset.")
                else:
                    print("Skipping entry due to extraction or processing failure.")
        
        elif choice == '2':
            size = get_dataset_size()
            print(f"Current dataset size: {size} records")
            if size < 20: #increased minimum slightly, adjust as needed
                print("Dataset too small for effective training. Please add more data (at least 20 records and 2 categories).")
                continue
            
            confirm_train = input("Are you sure you want to (re)train all models? (y/n): ").strip().lower()
            if confirm_train == 'y':
                print("Loading dataset for training...")
                try:
                    df = load_dataset()
                    if df.empty or len(df['category'].unique()) < 2:
                         print("Dataset is empty or has less than 2 unique categories. Training aborted.")
                         continue
                    print("Starting model training process...")
                    train_model(df) #this function now handles ensemble and prints accuracies
                    print("Model training completed.")
                except FileNotFoundError:
                     print("Dataset file not found. Please add data first.")
                except Exception as e:
                    print(f"An error occurred during training: {e}")
            else:
                print("Model training cancelled.")

        elif choice == '3':
            try:
                #check if models exist before attempting prediction
                _, _ = load_trained_models_and_vectorizer() 
            except FileNotFoundError:
                print("Models not trained yet. Please train models (Option 2) before predicting.")
                continue
            except Exception as e: #catch other loading errors
                print(f"Error loading models: {e}. Please retrain.")
                continue


            url_to_predict = input("Enter website URL to categorize: ").strip()
            if not url_to_predict:
                print("URL cannot be empty.")
                continue
            
            print("Predicting...")
            prediction, _ = predict_category_ensemble(url_to_predict) #we only need prediction here
            print(f"\nPredicted Category for {url_to_predict}: ** {str(prediction).upper()} **")

        elif choice == '4':
            size = get_dataset_size()
            print(f"Current dataset size: {size} records.")

        elif choice == '5':
            clear_prediction_cache()

        elif choice == '6':
            print("Exiting Website Categorizer CLI.")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_cli()
