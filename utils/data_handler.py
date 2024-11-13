import pandas as pd

def load_data(uploaded_file):
    """Load data from uploaded CSV file."""
    return pd.read_csv(uploaded_file)

