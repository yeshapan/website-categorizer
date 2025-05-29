# app.py — Streamlit UI for Website Categorization

import streamlit as st
from categorizer.model import predict_category

# -------------------------------
# 💡 Utility function: Set Page
# -------------------------------
def set_page_config():
    st.set_page_config(
        page_title="Website Categorizer",
        layout="wide"
    )

#sidebar config
def render_sidebar():
    with st.sidebar:
        st.title("Web Categorizer")
        st.markdown("A simple ML-powered tool that scrapes the web and classifies a website into categories based on its content.")

        st.markdown("---")
        st.markdown("**Built by:** Your Name / Team")
        st.markdown("**Model:** Logistic Regression + TF-IDF")
        st.markdown("**Tech Stack:** Python, Sklearn, Streamlit")
        st.markdown("")

#model interaction
def predict_section():
    st.subheader("Enter Website URL")
    url = st.text_input("Paste a valid website URL below", placeholder="https://example.com")

    if st.button("Predict Category"):
        if not url:
            st.warning("Please enter a website URL.")
            return

        try:
            with st.spinner("Extracting content and predicting..."):
                category, preview = predict_category(url)
                st.success(f"Predicted Category: **{category.capitalize()}**")

                with st.expander("View Extracted Text Preview"):
                    st.text_area(label="Cleaned Website Text (first 2000 characters)", value=preview, height=300)

        except Exception as e:
            st.error(f"An error occurred: {e}")

#UI
def render_main_layout():
    predict_section()

#main entry point
def main():
    set_page_config()
    render_sidebar()
    st.title("Website Category Predictor")
    st.markdown("Use this tool to determine the type of a website using natural language processing and machine learning.")
    render_main_layout()


if __name__ == "__main__":
    main()

