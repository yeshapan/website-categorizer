# Website Categorizer

A simple ML-powered web app that scrapes website content and classifies the site into relevant categories based on textual data.
Built using **Streamlit** for the frontend and a **Logistic Regression** model trained on web content for classification.
> The dataset created using scraping the web (and stemmed and cleaned text) for this project is included in this repo

> At present, this model has achieved 82.22% accuracy

## Features
-  Accepts any valid website URL
-  Extracts and cleans website text by scraping the web
-  Predicts the website category using machine learning
-  Displays a preview of extracted text for transparency
-  Fast and lightweight UI built with Streamlit
> Note that this project works best with English language websites

> here's a brief demo showing the working of this project on localhost: https://youtu.be/tKsbhTqBQqQ

## Folder Structure
``` bash
website-categorizer/
├── app/
│   └── app.py                  # Streamlit UI
├── categorizer/               
│   ├── __init__.py         #makes 'categorizer' a Python package (important for imports)
│   ├── model.py            #handles training, prediction, and management of multiple ML models
│   ├── dataset.py          #for loading and managing dataset
│   ├── scraper.py          #for fetching website content
│   ├── preprocess.py       #for cleaning/preparing text data
│   └── models/
│       ├── website_vectorizer.joblib           #TF-IDF vectorizer
│       ├── logistic_regression_model.joblib    #Logistic Regression model
│       ├── svm_model.joblib                    #SVM model
│       └── xgboost_model.joblib                #XGBoost model
├── .cache/                     #directory for caching (created by joblib.Memory, add to .gitignore)
├── main.py
├── websites_dataset.csv        #dataset
├── pyproject.toml              #Poetry dependency file
├── poetry.lock                 #Poetry lock file
├── Dockerfile                  #Instructions to build Docker image
├── .dockerignore               #specifies which files to ignore while building the Docker image
└── ReadME.md
```

## Steps to replicate the project:
### 1. Clone the repository
```bash
git clone https://github.com/yeshapan/website-categorizer.git
cd website-categorizer
```

### 2. Install Poetry (if not done already)
Follow official instructions: https://python-poetry.org/docs/#installation

### 3. Install dependencies via poetry
```bash
poetry install
```

### 4. Run the streamlit app
```bash
poetry run streamlit run app.py
```



