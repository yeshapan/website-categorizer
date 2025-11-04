# Website Categorizer

A lightweight simple ML-powered web app that scrapes website content and classifies the site into relevant categories based on textual data.
Built using **Streamlit** for the UI and an ensemble of **Logistic Regression** and **SVM** models (weighted stack: SVM-0.7 and LR-0.3) trained on web content for classification.
> The dataset created using scraping the web (and stemmed and cleaned text) for this project is included in this repo

> #### At present, this model has achieved **96.36% ensemble accuracy**

### Features
-  Accepts any valid website URL
-  Extracts and cleans website text by scraping the web
-  Predicts the website category using machine learning
-  Displays a preview of extracted text for transparency
-  Fast and lightweight UI built with Streamlit
> Note that this project works best with English language websites

> here's a brief demo video showing the working of this project on localhost: https://youtu.be/UgTwH9aM_Bo

### Folder Structure
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
│       └── svm_model.joblib                    #SVM model
│        
├── .cache/                     #directory for caching (created by joblib.Memory, add to .gitignore)
├── main.py
├── websites_dataset.csv        #dataset
├── pyproject.toml              #Poetry dependency file
├── poetry.lock                 #Poetry lock file
├── Dockerfile                  #Instructions to build Docker image
├── .dockerignore               #specifies which files to ignore while building the Docker image
└── ReadME.md
```

### Steps to replicate the project:
#### 1. Clone the repository
```bash
git clone https://github.com/yeshapan/website-categorizer.git
cd website-categorizer
```

#### 2. Install Poetry (if not done already)
Follow official instructions: https://python-poetry.org/docs/#installation

#### 3. Install dependencies via poetry
```bash
poetry install
```

#### 4. Run main.py (CLI)
```bash
poetry run python main.py
```
> You will have to run main.py and train the model before running Streamlit app

#### 5. Run the streamlit app
```bash
poetry run streamlit run app/app.py
```

**Note:** you may have to modify PATH for running app.py as it is not is project's root directory
For eg:
```bash
$env:PYTHONPATH="C:\Users\USER\desktop\website-categorizer" #modify as per path to local directory on your system"
```


### Overview of Project Pipeline

**Training Pipeline (Offline)**
> this is the process to train the models
* Load csv file into a pandas df
* Text Preprocessing 
    * Convert text to lowercase
    * Remove digits and punctuation
    * Remove NLTK English stopwords
    * Apply Porter Stemmer to normalize words
* Feature Extraction - TfidfVectorizer is fit on the preprocessed text dataset (learn the vocabulary and IDF weights) 
* Model Training:
    * Train - test split
    * GridSearchCV is used to find the optimal hyperparameters for both models
    * The best-performing versions of both LR and SVM saved to disk

**Inference Pipeline (Live)**
> this is the step-by-step process that runs when a user enters a URL into the Streamlit app:
* Input - user submits a URL via Streamlit UI
* Caching - The system checks if URL already stored in the .cache/ directory → cached prediction returned
* Scraping - If not cached, scraper.py fetches the URL's content using requests. BeautifulSoup is used to parse the HTML
* Preprocessing
* Feature Extraction
* Ensemble Prediction:
    * The saved .joblib models loaded
    * Both models generate prediction for the text vector
    * An ensemble rule is applied:
            * If the models disagree;  SVM's prediction prioritized as final result
            * If they agree → prediction is used
* Output- The final predicted category is returned and displayed to the user in the Streamlit app



