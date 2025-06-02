import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from scipy.stats import mode as majority_vote # For ensemble prediction
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

#define paths
MODEL_DIR = Path("categorizer/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True) #ensure directory exists

VECTORIZER_PATH = MODEL_DIR / "website_vectorizer.joblib"
LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression_model.joblib"
SVM_MODEL_PATH = MODEL_DIR / "svm_model.joblib"
XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost_model.joblib"

#model training
def train_model(df: pd.DataFrame):
    """
    Trains a TF-IDF vectorizer and multiple classification models (Logistic Regression, SVM, XGBoost).
    Saves the vectorizer and trained models.
    Reports accuracy for each model and the ensemble.
    """
    df = df.dropna(subset=['text', 'category'])
    if df.empty or len(df['category'].unique()) < 2:
        print("Not enough data or classes to train. Need at least 2 unique categories and some data.")
        return

    X = df["text"]
    y = df["category"]

    #TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), min_df=3, max_df=0.90)
    X_vec = vectorizer.fit_transform(X)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Vectorizer saved.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    #models definition with basic hyperparameter tuning setup (can be expanded)
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced"),
            "params": {"C": [5, 7.5, 10], "penalty": ["l1", "l2"]}
        },
        "SVM": {
            "model": LinearSVC(max_iter=1000, class_weight="balanced"), # probability=True for predict_proba if needed later
            "params": {"C": [0.1, 1, 10]}
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
        }
    }

    trained_models = {}
    test_predictions = {}

    for name, model_info in models.items():
        print(f"\nTraining {name}...")
        #using GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model_info["model"], model_info["params"], cv=3, scoring='accuracy', n_jobs=-1) # cv=3 due to small dataset
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        trained_models[name] = best_model
        
        #save individual model
        if name == "Logistic Regression":
            joblib.dump(best_model, LOGISTIC_MODEL_PATH)
        elif name == "SVM":
            joblib.dump(best_model, SVM_MODEL_PATH)
        elif name == "XGBoost":
            joblib.dump(best_model, XGBOOST_MODEL_PATH)
        print(f"{name} trained and saved with best params: {grid_search.best_params_}")

        y_pred = best_model.predict(X_test)
        test_predictions[name] = y_pred
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc * 100:.2f}%")
        # print(f"\nClassification Report for {name}:")
        # print(classification_report(y_test, y_pred, zero_division=0))


    #Ensemble Prediction (Majority Vote)
    #stack predictions from all models for the test set
    stacked_preds = pd.DataFrame(test_predictions).values
    
    #perform majority vote. `keepdims=True` is important for scipy > 1.9.0
    #result of majority_vote will be a tuple (mode_values, mode_counts)
    ensemble_pred_values, _ = majority_vote(stacked_preds, axis=1, keepdims=True)
    
    #ensemble_pred_values is a 2D array, need to flatten it to 1D for accuracy_score
    ensemble_pred_flat = ensemble_pred_values.flatten()

    ensemble_acc = accuracy_score(y_test, ensemble_pred_flat)
    print(f"\nEnsemble (Majority Vote) Accuracy: {ensemble_acc * 100:.2f}%")
    print("\nClassification Report for Ensemble:")
    print(classification_report(y_test, ensemble_pred_flat, zero_division=0))

    if not any([LOGISTIC_MODEL_PATH.exists(), SVM_MODEL_PATH.exists(), XGBOOST_MODEL_PATH.exists()]):
        print("No models were saved. Please check training logs.")

#Model Loading
def load_trained_models_and_vectorizer():
    """Loads all trained models and the TF-IDF vectorizer."""
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}. Please train the models first.")
    
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    models_to_load = {
        "Logistic Regression": LOGISTIC_MODEL_PATH,
        "SVM": SVM_MODEL_PATH,
        "XGBoost": XGBOOST_MODEL_PATH
    }
    
    loaded_models = {}
    for name, path in models_to_load.items():
        if path.exists():
            loaded_models[name] = joblib.load(path)
        else:
            print(f"Warning: Model file for {name} not found at {path}. It will be excluded from ensemble.")
            
    if not loaded_models:
        raise FileNotFoundError("No trained models found. Please train the models first.")
        
    return vectorizer, loaded_models

#Prediction Logic
#for caching, we'll use joblib.Memory
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = joblib.Memory(CACHE_DIR, verbose=0)

@memory.cache # This will cache the results of this function
def predict_category_ensemble(url: str):
    """
    Scrapes a URL, preprocesses the text, and predicts category using an ensemble of models.
    Uses caching to store and retrieve results for previously seen URLs.
    """
    #these imports are here to avoid circular dependencies if model.py is imported elsewhere
    #and to ensure they are available in the cached function's scope.
    from categorizer.scraper import scrape_website
    from categorizer.preprocess import preprocess_text

    print(f"Predicting for URL (cache {'hit' if memory.store.is_cached(['predict_category_ensemble', url]) else 'miss'}): {url}")

    raw_text = scrape_website(url)
    if not raw_text:
        return "Error: Could not scrape website", ""

    clean_text = preprocess_text(raw_text)
    if not clean_text:
        return "Error: Could not preprocess text", raw_text[:2000]

    try:
        vectorizer, models = load_trained_models_and_vectorizer()
    except FileNotFoundError as e:
        return f"Error: {e}", raw_text[:2000]

    if not models:
        return "Error: No models loaded for prediction.", raw_text[:2000]

    text_vec = vectorizer.transform([clean_text])
    
    predictions = []
    for model_name, model in models.items():
        try:
            pred = model.predict(text_vec)[0]
            predictions.append(pred)
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
            #optionally append a placeholder or handle error differently
    
    if not predictions:
        return "Error: All models failed to predict.", raw_text[:2000]

    #perform majority vote
    #majority_vote returns (array_of_modes, array_of_counts)
    final_prediction_array, _ = majority_vote(predictions, keepdims=False) # keepdims=False for single value
    final_prediction = final_prediction_array #it's already the single most common item

    return final_prediction, raw_text[:2000]

def clear_prediction_cache():
    """Clears the prediction cache."""
    try:
        memory.clear(warn=False)
        print("Prediction cache cleared.")
    except Exception as e:
        print(f"Error clearing cache: {e}")
