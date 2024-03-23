import pandas as pd
import streamlit as st
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


def csv_prediction():
    # Load ML models
    pickle_file_path = 'random_forest_updated.pkl'
    with open(pickle_file_path, 'rb') as file:
        loaded_model = pickle.load(file)

    pickle_file_path = 'xgboost_regression.pkl'
    with open(pickle_file_path, 'rb') as file:
        reg_model = pickle.load(file)

    # Main title
    st.title("MACHINE MAINTENANCE SCHEDULING")

    # File uploader in sidebar
    st.sidebar.title("Upload CSV Data File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    def read_dataframe(file):
        if file is not None:
            df = pd.read_csv(file)
            st.data_editor(df)


    # Function to process CSV file and show results
    def process_csv_file(file):
        if file is not None:
            # Read CSV file
            df = pd.read_csv(file)

            # Make predictions
            predicted_class = loaded_model.predict(df)
            failure_probability = loaded_model.predict_proba(df)
            predicted_rul = reg_model.predict(df)

            # Add predictions to DataFrame
            df['Output'] = predicted_class
            df['Failure_Probability'] = failure_probability[:, 1] * 100
            df['RUL'] = predicted_rul

            # Show DataFrame
            with st.form("my_form"):
                st.header("REAL TIME FEEDBACK AND MODEL RETRAINING")
                edited_df = st.data_editor(df,num_rows='dynamic')
                submitted = st.form_submit_button("Retrain the model")

                if submitted:
                    # Retraining the model on edited data
                    X = edited_df.drop(['Output','Failure_Probability','RUL'],axis=1)
                    y = edited_df['Output']
                    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
                    y_reg = edited_df['RUL']
                    X_reg_train,X_reg_test,y_reg_train,y_reg_test = train_test_split(X,y_reg,test_size=0.2,random_state=42)
                    # Classification model Retraining
                    st.header("Classification Model Retrained Metrics")
                    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False,
                                                eval_metric='logloss', n_jobs=-1, n_estimators=100, eta=0.1)
                    xgb_model.fit(X_train,y_train)
                    y_pred = loaded_model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(label="Accuracy", value=f"{accuracy:.2f}")

                    with col2:
                        st.metric(label="Precision", value=f"{precision:.2f}")

                    with col3:
                        st.metric(label="Recall", value=f"{recall:.2f}")

                    with col4:
                        st.metric(label="F1 Score", value=f"{f1:.2f}")

                    st.header("Regression Model Retrained Metrics")
                    rf = RandomForestRegressor(oob_score=True, random_state=42,n_estimators=100)
                    rf.fit(X_reg_train, y_reg_train)
                    predictions = rf.predict(X_reg_test)
                    r2 = r2_score(y_reg_test, predictions)
                    oob_score = rf.oob_score_
                    col5, col6= st.columns(2)
                    with col5:
                        st.metric(label="R2 SCORE", value=f"{r2:.2f}")
                    with col6:
                        st.metric(label="OOB SCORE", value=f"{oob_score:.2f}")

    # Check if submit button is clicked
    if st.button("Submit"):
        st.session_state['button_pressed'] = True

    if st.session_state.get("button_pressed", False):
        process_csv_file(uploaded_file)
csv_prediction()