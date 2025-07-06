import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import smtplib
import ssl
from langchain.prompts import HumanMessagePromptTemplate,ChatPromptTemplate,SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI


def model_run():
    # Assuming your XGBoost model and any necessary preprocessing tools are loaded correctly
    pickle_file_path = 'random_forest_updated.pkl'
    with open(pickle_file_path, 'rb') as file:
        loaded_model = pickle.load(file)

    pickle_file_path = 'xgboost_regression.pkl'
    with open(pickle_file_path, 'rb') as file:
        reg_model = pickle.load(file)

    # Function to send email (unchanged)
    def send_email(sender_email, receiver_email, subject, message, email_password):
        smtp_server = "smtp.gmail.com"
        port = 587  # For starttls

        # Create a secure SSL context
        context = ssl.create_default_context()

        # Try to log in to server and send email
        with smtplib.SMTP(smtp_server, port) as server:
            server.ehlo()  # Can be omitted
            server.starttls(context=context)  # Secure the connection
            server.ehlo()  # Can be omitted
            server.login(sender_email, email_password)
            email_content = f"Subject: {subject}\n\n{message}"
            server.sendmail(sender_email, receiver_email, email_content)
        st.success("Email sent successfully!")


    st.title("ML+LLM Decision System")
    with st.form(key='machine_input_form'):
        machine_type = st.selectbox('Machine Type', options=['L', 'M', 'H'], help='Select the type of machine (L, M, H)')
        air_temperature = st.number_input('Air Temperature', format="%.2f", help='Enter the air temperature (float value)',
                                        value=298.5)
        process_temperature = st.number_input('Process Temperature', format="%.2f",
                                            help='Enter the process temperature (float value)', value=308.5)
        rotational_speed = st.number_input('Rotational Speed', format="%.2f",
                                        help='Enter the rotational speed (float value)', value=1500.5)
        torque = st.number_input('Torque', format="%.2f", help='Enter the torque (float value)', value=40.5)
        tool_wear = st.number_input('Tool Wear', format="%.2f", help='Enter the tool wear (float value)', value=0.5)
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        power = torque * rotational_speed
        # Map machine type to numeric value
        type_mapping = {'L': 0, 'M': 1, 'H': 2}
        machine_type_numeric = type_mapping[machine_type]

        # Assuming the model expects the features in a specific order
        features = np.array(
            [[machine_type_numeric, air_temperature, process_temperature, rotational_speed, torque, tool_wear, power]])

        # Predict with the model
        predicted_class = loaded_model.predict(features)
        failure_proba = loaded_model.predict_proba(features)
        probability = failure_proba[0,1]*100
        # Display the prediction result
        st.header("Estimated Output Class")
        if predicted_class == 1.0:
            st.write(":orange[Machine Failure Detected]")

            # Real time Notification
            sender_email = "enter-email-id"  # Replace with your email
            receiver_email = "enter-email-id"  # Replace with recipient's email
            subject = "Machine Failed Notification"
            message = f"Machine can be failed according to ML Model,Urgent Attention Needed"
            email_password = "enter-email-password"  # Replace with your email password
            send_email(sender_email, receiver_email, subject, message, email_password)
        else:
            st.write(":green[No Machine Failure Detected]")

        #REGRESSION RUL PREDICTION
        reg_features = np.array(
            [[machine_type_numeric, air_temperature, process_temperature, rotational_speed, torque, tool_wear, power]])
        predicted_rul = reg_model.predict(reg_features)
        st.header("Estimated RUL Time in Minutes")
        st.write(predicted_rul)

        ## LLM POWERED DECISION SYSTEM
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    """ We have trained 2 models,One for Classification and another one Regression.
                        The Dataset,I have used is Milling Machine Dataset for Machine Failure Prediction Modelling(Classification) 
                        and On same Dataset for RUL(Remaining Useful Life ) Estimation(Regression).
                        Dataset Columns Names and Description:
                        1)type: just the product type L, M or H from column 2.
                        2)air temperature [K]: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K.
                        3)process temperature [K]: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
                        4)rotational speed [rpm]: calculated from a power of 2860 W, overlaid with a normally distributed noise.
                        5)torque [Nm]: torque values are normally distributed around 40 Nm with a SD = 10 Nm and no negative values.
                        6)Regression Model Output Column(tool wear [min]): (breakdown and gradual failure of a cutting tool due to regular operation) The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.
                        7)Classification Model Output Column(A 'machine failure') label that indicates, whether the machine has failed in this particular datapoint for any of the following failure modes are true.
    
                        i)Classification Output Understanding as below:
                        Classification Output=1 than Machine has failed,0 means machine has not failed.
                        Probability of Machine Failure is given in decimal Format if machine is failed,Higher the decimal value,more probable is Machine to fail 
                        ii)Regression Output Understanding as Below:
                        Regression Output is Tool Wear Value in Minutes.
    
                        I will give you the Outputs of Both classification and regression model on the basis of This you have to to give Output in below format,
    
                        Output Format should be in Markdown Format:
                        1)You have to find uncertainty estimates in point wise format according to Outputs and Probability of Machine Failure.
                        2)Suggest maintenance decisions in point wise formaT according to your large knowledge about Milling Machine maintenance
            ...
        """

                ),
                HumanMessagePromptTemplate.from_template(
                    "Classification Output:{cls},Regression Output:{reg},Probability of Machine Failure:{proba}"),
            ]
        )
        messages = chat_template.format_messages(cls=predicted_class, reg=predicted_rul, proba=probability)
        chat = ChatOpenAI(openai_api_key="ENTER-OPENAI-API-KEY")
        response = chat(messages)
        st.header(":orange[LLM Based Decision System]")
        st.write(response.content)
model_run()

