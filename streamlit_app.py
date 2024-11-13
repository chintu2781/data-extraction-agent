import streamlit as st
import pandas as pd
from utils.data_handler import load_data
from utils.search_agent import search_web_and_extract_info

# Sidebar for file upload and Google Sheets connection
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Target column selection and query template
if uploaded_file:
    data = load_data(uploaded_file)
    st.write("Data Preview", data.head())
    column = st.selectbox("Select column for search queries", data.columns)
    query_template = st.text_input("Enter query template (e.g., 'Find the email for {Company}')")

    # Run searches
    if st.button("Run Searches"):
        results_df = search_web_and_extract_info(data, column, query_template)
        st.write("Extracted Results", results_df)
        csv = results_df.to_csv(index=False)
        st.download_button(label="Download Results as CSV", data=csv, mime="text/csv")

