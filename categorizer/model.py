import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #for heatmap

#define paths
MODEL_DIR = Path("categorizer/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True) #ensure directory exists

VECTORIZER_PATH = MODEL_DIR / "website_vectorizer.joblib"
LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression_model.joblib"
SVM_MODEL_PATH = MODEL_DIR / "svm_model.joblib"
RF_MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"

#weights for Ensemble (Based on individual accuracy)
MODEL_WEIGHTS = {
    "SVM": 0.9825,
    "Random Forest": 0.9825,
    "Logistic Regression": 0.9649
}

#model training
def train_model(df: pd.DataFrame):
    """
    Trains TF-IDF, LR, SVM, and Random Forest.
    Saves the vectorizer and trained models.
    Reports accuracy for each model and the Weighted Ensemble.
    Generates and displays a heatmap of the confusion matrix for the ensemble model.
    """
    df = df.dropna(subset=['text', 'category'])
    if df.empty or len(df['category'].unique()) < 2:
        print("Not enough data or classes to train. Need at least 2 unique categories and some data.")
        return

    X = df["text"]
    y = df["category"]

    #TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 3), min_df=3, max_df=0.90)
    X_vec = vectorizer.fit_transform(X)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Vectorizer saved.")

    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    #models definition with more extensive hyperparameter tuning
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced"),
            "params": {"C": [0.1, 1, 5, 7.5, 10, 20], "penalty": ["l1", "l2"]}
        },
        "SVM": {
            "model": LinearSVC(dual="auto", max_iter=2000, class_weight="balanced"),
            "params": {"C": [0.01, 0.1, 1, 10, 100]}
        },
        "Random Forest": { 
            "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "params": {
                'n_estimators': [200, 300],
                'max_depth': [20, 30, None],
                'min_samples_split': [2, 5],
                'bootstrap': [False, True]
            }
        }
    }

    trained_models = {}
    test_predictions = {} #to store string predictions for all models

    for name, model_info in models.items():
        print(f"\nTraining {name}...")
        grid_search = GridSearchCV(model_info["model"], model_info["params"], cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        test_predictions[name] = y_pred #store string predictions
        trained_models[name] = best_model

        if name == "Logistic Regression": joblib.dump(best_model, LOGISTIC_MODEL_PATH)
        elif name == "SVM": joblib.dump(best_model, SVM_MODEL_PATH)
        elif name == "Random Forest": joblib.dump(best_model, RF_MODEL_PATH)

        print(f"{name} trained and saved with best params: {grid_search.best_params_}")
        print(f"{name} Accuracy: {acc * 100:.2f}%")


    #ensemble prediction logic (Weighted Voting)
    if test_predictions:
        ensemble_type_message = "Ensemble (Weighted Voting)"
        print(f"\nCalculating {ensemble_type_message}...")

        ensemble_final_predictions_list = []
        #iterate through test set index-wise
        for i in range(len(y_test)):
            scores = {}
            for name, preds in test_predictions.items():
                pred_cat = preds[i]
                weight = MODEL_WEIGHTS.get(name, 1.0)
                scores[pred_cat] = scores.get(pred_cat, 0) + weight
            
            #find category with highest score
            #in case of exact tie (SVM vs RF), this picks the one with highest sort order (effectively random but stable)
            best_cat = max(scores, key=scores.get)
            ensemble_final_predictions_list.append(best_cat)
        
        ensemble_pred_flat = np.array(ensemble_final_predictions_list)
        ensemble_acc = accuracy_score(y_test, ensemble_pred_flat)
        print(f"\n{ensemble_type_message} Accuracy: {ensemble_acc * 100:.2f}%")
        print(f"\nClassification Report for {ensemble_type_message}:")
        report_labels = sorted(list(set(y_test) | set(ensemble_pred_flat)))
        print(classification_report(y_test, ensemble_pred_flat, labels=report_labels, zero_division=0))

        # Generate and display confusion matrix heatmap
        cm = confusion_matrix(y_test, ensemble_pred_flat, labels=report_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=report_labels, yticklabels=report_labels)
        plt.title(f"Confusion Matrix for {ensemble_type_message}")
        plt.ylabel("Actual Category")
        plt.xlabel("Predicted Category")
        plt.show()

    else:
        print("\nNo model predictions available for ensemble.")

    if not any([LOGISTIC_MODEL_PATH.exists(), SVM_MODEL_PATH.exists(), RF_MODEL_PATH.exists()]):
        print("No models were saved. Please check training logs.")

#model Loading
def load_trained_models_and_vectorizer():
    """Loads all trained models and the TF-IDF vectorizer."""
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}. Please train the models first.")
    vectorizer = joblib.load(VECTORIZER_PATH)

    models_to_load = {
        "Logistic Regression": LOGISTIC_MODEL_PATH,
        "SVM": SVM_MODEL_PATH,
        "Random Forest": RF_MODEL_PATH
    }
    loaded_models = {}
    for name, path in models_to_load.items():
        if path.exists():
            loaded_models[name] = joblib.load(path)
        else:
            print(f"Warning: Model file for {name} not found at {path}. It will be excluded from ensemble.")

    return vectorizer, loaded_models

#Prediction Logic
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)

@memory.cache
def predict_category_ensemble(url: str):
    """
    Scrapes a URL, preprocesses the text, and predicts category.
    Uses Weighted Voting (SVM, RF, LR) to determine the best category
    Uses caching to store and retrieve results for previously seen URLs.
    """
    from categorizer.scraper import scrape_website #local import for caching
    from categorizer.preprocess import preprocess_text #local import for caching

    print(f"Predicting for URL: {url}")

    raw_text = scrape_website(url)
    if not raw_text:
        return "Error: Could not scrape website", ""

    clean_text = preprocess_text(raw_text)
    preview_text = raw_text[:2000] #keep original raw text for preview
    if not clean_text:
        return "Error: Could not preprocess text", preview_text

    try:
        vectorizer, models = load_trained_models_and_vectorizer()
    except FileNotFoundError as e:
        return f"Error: {e}", preview_text
    except Exception as e: #catch any other loading error
        return f"Error loading models/vectorizers: {e}", preview_text

    if not models:
        return "Error: No models loaded for prediction.", preview_text

    text_vec = vectorizer.transform([clean_text])
    
    #weighted voting logic
    scores = {}
    
    for name, model in models.items():
        try:
            pred = str(model.predict(text_vec)[0])
            weight = MODEL_WEIGHTS.get(name, 0)
            scores[pred] = scores.get(pred, 0) + weight
        except Exception as e:
            print(f"Warning: Error predicting with {name}: {e}")

    if not scores:
         return "Error: No valid predictions made.", preview_text

    #sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    #check for tie at the top (e.g. SVM vs RF with equal weights)
    #tie-breaking rule: Prefer Random Forest if it's involved in the tie
    final_prediction = sorted_scores[0][0]
    
    if len(sorted_scores) > 1 and sorted_scores[0][1] == sorted_scores[1][1]:
        # Tie detected. Get RF prediction specifically if available
        rf_pred = None
        if "Random Forest" in models:
             try:
                 rf_pred = str(models["Random Forest"].predict(text_vec)[0])
             except: pass
        
        #if RF prediction exists and is one of the tied winners, choose it
        if rf_pred and rf_pred in [sorted_scores[0][0], sorted_scores[1][0]]:
             final_prediction = rf_pred

    return str(final_prediction), preview_text

def clear_prediction_cache():
    """Clears the prediction cache."""
    try:
        memory.clear(warn=False)
        print("Prediction cache cleared.")
    except Exception as e:
        print(f"Error clearing cache: {e}")