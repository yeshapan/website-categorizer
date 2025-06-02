#Streamlit UI

import streamlit as st
#no sys.path manipulation needed if app is run from root or Docker context is correct.
from categorizer.model import predict_category_ensemble, clear_prediction_cache, load_trained_models_and_vectorizer
from categorizer.dataset import get_dataset_size #to show dataset stats (optional) 
import os #for checking model existence

#check if models exist to provide user feedback
MODELS_EXIST = False
try:
    #attempt to load to see if they are present and valid
    load_trained_models_and_vectorizer()
    MODELS_EXIST = True
except FileNotFoundError:
    MODELS_EXIST = False
except Exception: #catch other potential load errors (e.g. corrupted files)
    MODELS_EXIST = False


def add_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background-color: white;
            color: #333333;
        }
        [data-testid="stSidebar"] {
            background-color: #f0f0f0;
            color: #333333;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #ff7e5f, #feb47b, #ff6a95); /* Example gradient */
            border: none;
            color: white;
            font-weight: 600;
            padding: 10px 24px;
            border-radius: 8px;
        }
        div.stButton > button:hover {
            opacity: 0.8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_page_config():
    st.set_page_config(
        page_title="Website Categorizer",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar():
    with st.sidebar:
        st.title("Web Categorizer")
        st.markdown("Classify websites into categories using an ensemble of ML models and NLP.")
        st.markdown("---")
        st.markdown("### Models Used:")
        st.markdown("- Logistic Regression\n- Support Vector Machine (SVM)\n- XGBoost Classifier")
        st.markdown("---")
        dataset_size = get_dataset_size()
        st.metric("Dataset Size (records)", dataset_size)
        st.markdown("---")
        if st.button("Clear Prediction Cache"):
            clear_prediction_cache()
            st.success("Cache cleared!")
            st.rerun() #rerun to update UI if needed

        st.markdown("---")
        st.markdown("Built by Yesha Pandya")
        st.markdown("@coffee.compile")


def predict_section():
    st.header("Predict Website Category")
    url = st.text_input("Enter a full website URL (e.g., https://www.example.com)", placeholder="https://example.com")

    if st.button("Predict Category", type="primary"):
        if not MODELS_EXIST:
            st.error("Models not found or not trained yet. Please train the models using the CLI (`poetry run python main.py`, option 2).")
            return

        if not url:
            st.warning("Please enter a website URL.")
            return
        
        if not (url.startswith("http://") or url.startswith("https://")):
            st.warning("Please enter a valid URL starting with http:// or https://")
            return

        try:
            with st.spinner("Scraping content and making a prediction... Hold tight!"):
                #predict_category_ensemble is already cached with joblib.Memory
                #Streamlit's caching is an alternative, but joblib is already set up
                category, preview = predict_category_ensemble(url)
            
            if "Error:" in category:
                 st.error(f"Prediction failed: {category}")
            else:
                st.success(f"Predicted Category: **{str(category).upper()}**")

            if preview:
                with st.expander("View Extracted Text Preview (first 2000 characters)"):
                    st.text_area(label="Text preview:", value=preview, height=250, disabled=True)

        except FileNotFoundError as e:
             st.error(f"Critical error: Model files or vectorizer not found. {e}. Please ensure models are trained via CLI.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.exception(e) # Shows full traceback for debugging

def render_main_layout():
    if not MODELS_EXIST:
        st.warning(
            """
            **Welcome! The machine learning models seem to be missing.**

            To enable predictions, please:
            1. Add data to your `websites_dataset.csv`.
            2. Train the models by running the command line tool:
               Open your terminal in the project root and execute:
               `poetry run python main.py`
               Then choose option '2' to train.
            """
        )
    else:
        st.info("✨ Models are loaded! Ready to predict.")
    
    predict_section()


def main_streamlit():
    set_page_config()
    add_custom_css() #call custom CSS function
    render_sidebar()
    
    st.title("Website Category Predictor Dashboard")
    st.markdown("Leverage the power of NLP and ensemble machine learning to classify any website!")
    st.markdown("---")

    render_main_layout()

if __name__ == "__main__":
    main_streamlit()