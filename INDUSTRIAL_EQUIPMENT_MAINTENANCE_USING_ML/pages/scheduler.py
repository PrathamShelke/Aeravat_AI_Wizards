import streamlit as st
import pandas as pd

def scheduler():
    st.title(":rainbow[Machine maintenance Scheduler]")

    # File uploader
    st.sidebar.title(":rainbow[Upload CSV File]")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Show DataFrame
        st.write("DataFrame Preview")
        st.snow()
        st.data_editor(df)
scheduler()
