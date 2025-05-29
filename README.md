# Website Categorizer

A simple yet powerful ML-powered web app that scrapes website content and classifies the site into relevant categories based on textual data.
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

## Folder Structure
``` bash
website-categorizer/
├── app/
│   └── app.py                  # Streamlit UI
├── categorizer/                # Model, scraping, and preprocessing logic
│   ├── model.py
│   ├── dataset.py
│   ├── scraper.py
│   └── preprocess.py
├── main.py
├── websites_dataset.csv        # Sample dataset
├── pyproject.toml              # Poetry dependency file
├── poetry.lock
├── README.md                   # This file
└── .gitignore
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



