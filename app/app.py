#streamlit UI

import streamlit as st
from categorizer.model import predict_category
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Set overall white background */
        .main {
            background-color: white;
            color: #333333
        }

        /* Sidebar background light grey */
        [data-testid="stSidebar"] {
            background-color: #f0f0f0;
            color: #333333
        }

        div.stButton > button {
            background: linear-gradient(90deg, #ff7e5f, #feb47b, #ff6a95);
            border: none;
            color: white;
            font-weight: 600;
        </style>

        """,
        unsafe_allow_html=True,
    )

#set page is a utility function
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
        st.markdown("ML model used: *Logistic Regression*")
        st.markdown("@coffee.compile")

#model interaction
def predict_section():
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
    add_custom_css()
    render_main_layout()


if __name__ == "__main__":
    main()

