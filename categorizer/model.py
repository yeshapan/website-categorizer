import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
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

#model training
def train_model(df: pd.DataFrame):
    """
    Trains a TF-IDF vectorizer and two classification models (Logistic Regression, SVM).
    Saves the vectorizer and trained models.
    Reports accuracy for each model and the ensemble.
    Ensemble prioritizes SVM if Logistic Regression and SVM disagree.
    Generates and displays a heatmap of the confusion matrix for the ensemble model.
    """
    df = df.dropna(subset=['text', 'category'])
    if df.empty or len(df['category'].unique()) < 2:
        print("Not enough data or classes to train. Need at least 2 unique categories and some data.")
        return

    X = df["text"]
    y = df["category"]

    #TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=3, max_df=0.90)
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
        }
    }

    trained_models = {}
    test_predictions = {} #to store string predictions for all models

    for name, model_info in models.items():
        print(f"\nTraining {name}...")
        grid_search = GridSearchCV(model_info["model"], model_info["params"], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        test_predictions[name] = y_pred #store string predictions

        trained_models[name] = best_model

        model_path = None
        if name == "Logistic Regression": model_path = LOGISTIC_MODEL_PATH
        elif name == "SVM": model_path = SVM_MODEL_PATH

        if model_path:
            joblib.dump(best_model, model_path)
        print(f"{name} trained and saved with best params: {grid_search.best_params_}")
        print(f"{name} Accuracy: {acc * 100:.2f}%")


    # Ensemble Prediction Logic
    ensemble_type_message = ""

    if "Logistic Regression" in test_predictions and "SVM" in test_predictions:
        ensemble_type_message = "Ensemble (Weighted Choice: SVM 0.7, LR 0.3)"
        print(f"\nCalculating {ensemble_type_message}...")
        
        preds_lr = test_predictions["Logistic Regression"]
        preds_svm = test_predictions["SVM"]
        
        ensemble_final_predictions_list = []
        for i in range(len(y_test)):
            lr_pred_sample = preds_lr[i]
            svm_pred_sample = preds_svm[i]

            if lr_pred_sample == svm_pred_sample:
                ensemble_final_predictions_list.append(lr_pred_sample)
            else:
                # SVM prediction (higher weight) wins in case of disagreement
                ensemble_final_predictions_list.append(svm_pred_sample)
        
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

    elif test_predictions: #fallback if the specific LR/SVM combo wasn't available, but other predictions exist
        ensemble_type_message = "Ensemble (Majority Vote - Fallback)"
        print(f"\nBoth Logistic Regression and SVM predictions not simultaneously available for weighted choice.")
        print(f"Performing standard Majority Vote on available model predictions ({ensemble_type_message})...")
        
        stacked_preds_df = pd.DataFrame(test_predictions)
        if not stacked_preds_df.empty:
            ensemble_pred_flat = stacked_preds_df.mode(axis=1).iloc[:, 0].values
            ensemble_acc = accuracy_score(y_test, ensemble_pred_flat)
            print(f"\n{ensemble_type_message} Accuracy: {ensemble_acc * 100:.2f}%")
            print(f"\nClassification Report for {ensemble_type_message}:")
            report_labels_fallback = sorted(list(set(y_test) | set(ensemble_pred_flat)))
            print(classification_report(y_test, ensemble_pred_flat, labels=report_labels_fallback, zero_division=0))

            # Generate and display confusion matrix heatmap for fallback ensemble
            cm = confusion_matrix(y_test, ensemble_pred_flat, labels=report_labels_fallback)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=report_labels_fallback, yticklabels=report_labels_fallback)
            plt.title(f"Confusion Matrix for {ensemble_type_message}")
            plt.ylabel("Actual Category")
            plt.xlabel("Predicted Category")
            plt.show()

        else:
            print("\nNo predictions available in test_predictions for fallback majority vote.")
    else:
        print("\nNo model predictions available for ensemble.")


    if not any([LOGISTIC_MODEL_PATH.exists(), SVM_MODEL_PATH.exists()]):
        print("No models were saved. Please check training logs.")

#Model Loading
def load_trained_models_and_vectorizer():
    """Loads all trained models and the TF-IDF vectorizer."""
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}. Please train the models first.")
    vectorizer = joblib.load(VECTORIZER_PATH)

    models_to_load = {
        "Logistic Regression": LOGISTIC_MODEL_PATH,
        "SVM": SVM_MODEL_PATH
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
    If Logistic Regression and SVM both predict, SVM's choice is prioritized in case of disagreement.
    Otherwise, uses a general majority vote of available model predictions.
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
    final_prediction = None
    
    # Attempt weighted choice (SVM priority) if both LR and SVM are loaded
    if "Logistic Regression" in models and "SVM" in models:
        pred_lr, pred_svm = None, None
        lr_model_available, svm_model_available = True, True

        try:
            pred_lr = str(models["Logistic Regression"].predict(text_vec)[0])
        except Exception as e:
            lr_model_available = False
            print(f"Warning: Error predicting with Logistic Regression for URL {url}: {e}. It will be excluded from this attempt.")
        
        try:
            pred_svm = str(models["SVM"].predict(text_vec)[0])
        except Exception as e:
            svm_model_available = False
            print(f"Warning: Error predicting with SVM for URL {url}: {e}. It will be excluded from this attempt.")

        if lr_model_available and svm_model_available and pred_lr is not None and pred_svm is not None:
            if pred_lr == pred_svm:
                final_prediction = pred_lr
            else:
                final_prediction = pred_svm # SVM wins
            print("Used weighted choice (LR/SVM with SVM priority) for prediction.")
        elif svm_model_available and pred_svm is not None:
            final_prediction = pred_svm
            print("Used SVM prediction (LR failed or was unavailable).")
        elif lr_model_available and pred_lr is not None:
            final_prediction = pred_lr
            print("Used Logistic Regression prediction (SVM failed or was unavailable).")
    
    # Fallback to original majority vote if weighted choice wasn't made or wasn't applicable
    if final_prediction is None:
        if "Logistic Regression" in models and "SVM" in models :
            print("Weighted choice did not yield a result, falling back to general majority vote for all available models.")
        else:
            print("Both Logistic Regression and SVM not available for weighted choice, using general majority vote.")

        predictions = []
        for model_name, model in models.items():
            try:
                pred = model.predict(text_vec)[0]
                predictions.append(str(pred))
            except Exception as e:
                print(f"Error predicting with {model_name} for URL {url} during fallback: {e}")
        
        if not predictions:
            return "Error: All models failed to predict or no valid predictions were made (fallback).", preview_text

        final_prediction_series = pd.Series(predictions).mode()

        if not final_prediction_series.empty:
            final_prediction = final_prediction_series[0]
        elif predictions:
            final_prediction = predictions[0]
        else:
            return "Error: No predictions available to determine final category (fallback).", preview_text
    
    if final_prediction is None:
        return "Error: Could not determine a final prediction for the URL.", preview_text
        
    return str(final_prediction), preview_text

def clear_prediction_cache():
    """Clears the prediction cache."""
    try:
        memory.clear(warn=False)
        print("Prediction cache cleared.")
    except Exception as e:
        print(f"Error clearing cache: {e}")