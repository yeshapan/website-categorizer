#using Naive Bayes for now

from sklearn.feature_extraction.text import TfidfVectorizer #to convert text to tf-idf feature matrix
from sklearn.naive_bayes import MultinomialNB #niave bayes classifier for text data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

def train_model(df):
    X = df["text"]
    X=X.dropna()
    y = df["category"]

    vectorizer = TfidfVectorizer(max_features=5000) #convert text to nums to feed ML model (only 5000 most imp words used to prevent overfitting)
    #Tf: term frequency (frequency of any word)
    #idf: Inverse document frequency (for unique/imp any word is across all docs)
    X_vec = vectorizer.fit_transform(X)

    #train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    
    '''#NB classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)'''

    #Logistic regressor
    model=LogisticRegression(max_iter=1000, solver= "liblinear", C=5.0, penalty= "l1", class_weight="balanced")
    #C is to tune regularization
    '''C=1.0 gives 65.71% accuracy ; C=2.0 gives 71.43% accuracy ; C=5.0 gives 82.86% accuracy ; C=8.0 gives 80.00% accuracy'''
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Model Accuracy: {acc * 100:.2f}%")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, vectorizer
