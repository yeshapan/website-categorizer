from sklearn.feature_extraction.text import TfidfVectorizer #to convert text to tf-idf feature matrix
#from sklearn.naive_bayes import MultinomialNB #niave bayes classifier for text data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib
import os

model_path = "categorizer/website_categorizer_model.joblib"
vectorizer_path = "categorizer/website_vectorizer.joblib"

def train_model(df):
    X = df["text"]
    X=X.dropna()
    y = df["category"]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), max_df=0.9) #convert text to nums to feed ML model (only 5000 most imp words used to prevent overfitting)
    #Tf: term frequency (frequency of any word)
    #idf: Inverse document frequency (for unique/imp any word is across all docs)
    X_vec = vectorizer.fit_transform(X)

    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    '''#NB classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)'''

    '''#Logistic regressor
    model=LogisticRegression(max_iter=1000, solver= "liblinear", C=7.5, penalty= "l1", class_weight="balanced")
    model.fit(X_train, y_train)
    #C is to tune regularization'''
    
    '''#Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, max_depth=None, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)'''

    svm_pipeline = Pipeline([
    ('clf', LinearSVC(C=1.0))
    ])

    svm_pipeline.fit(X_train, y_train)
    svm_accuracy = svm_pipeline.score(X_test, y_test)
    print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Model Accuracy: {acc * 100:.2f}%")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    #return model, vectorizer

    save_model(model, vectorizer)

def save_model(model, vectorizer):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Model and vectorizer saved")

def load_model():
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("file or vectorizer not found: pls train the model first")
    model= joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_category(url: str):
    from categorizer.scraper import scrape_website
    from categorizer.preprocess import preprocess_text

    raw_text = scrape_website(url)
    clean_text = preprocess_text(raw_text)

    model, vectorizer = load_model()
    X = vectorizer.transform([clean_text])
    prediction = model.predict(X)[0]

    return prediction, raw_text[:2000]
