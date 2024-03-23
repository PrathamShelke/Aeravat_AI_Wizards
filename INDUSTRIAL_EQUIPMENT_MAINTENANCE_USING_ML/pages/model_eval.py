import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def model_evaluation():
    data = pd.read_csv('Balanced_data.csv')

    st.title(":orange[MODEL EVALUATION METRIC]")

    st.subheader(":green[CLASSIFICATION MODEL METRICS]")
    cls = pd.read_excel('CLS.xlsx',engine='openpyxl')
    st.dataframe(cls)

    st.subheader(":green[REGRESSION MODEL METRICS]")
    reg = pd.read_excel('REG.xlsx',engine='openpyxl')
    st.dataframe(reg)

    # EDA SECTION
    st.header(":green[EDA on Historical Data]")
    variables = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
    fig, ax = plt.subplots(len(variables), 1, figsize=(8, 20))  # Changed to the correct usage of plt.subplots

    for i, var in enumerate(variables):
        sns.histplot(data=data, x=var, kde=True, ax=ax[i])
        ax[i].set_xlabel(var)
        ax[i].set_ylabel('Count')

    plt.tight_layout()
    st.pyplot(fig)

    # Correlation Heatmap
    st.header(':green[Correlation Heatmap]')
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)

    col1, col2 = st.columns(2)

    with col1:
        failure_types = data.loc[:, ['Machine failure']]
        rows_sum = failure_types.sum(axis=1)

        st.subheader(':green[Count of different failure types]')
        st.bar_chart(rows_sum.value_counts())
model_evaluation()
