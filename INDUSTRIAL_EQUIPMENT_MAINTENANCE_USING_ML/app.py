# app.py

import streamlit as st
import sqlite3
import hashlib

st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)

# Main function
def main():
    # Check if the user is logged in
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_signup()

# Login and signup function
def login_signup():
    st.subheader("Login / Sign Up")

    with st.container():
        st.markdown(
            """
            <style>
            .login-signup-box {
                margin-left: 40%;
                margin-right: auto;
                width: 50%; /* Adjust the width as needed */
                padding: 20px;
                border: 1px solid #ccc;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    login_option = st.radio("Choose an option", ("Login", "Sign Up"))

    if login_option == "Login":
        login()
    elif login_option == "Sign Up":
        signup()
        
    st.write("</div>", unsafe_allow_html=True)

# Login function
def login():
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    login_clicked = st.button("Login")  # Store the click event of the Login button

    if login_clicked:  # Check if the Login button is clicked
        if not username or not password:
            st.error("Please enter both username and password")
        else:
            hashed_password = hash_password(password)
            stored_password = get_stored_password(username)
            if stored_password is not None and stored_password == hashed_password:
                st.session_state.logged_in = True
                st.success("Successfully logged in as {}".format(username))
            else:
                st.error("Invalid username or password")

# Sign up function
def signup():
    st.subheader("Sign Up")

    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if new_password == confirm_password:
            if create_user(new_username, new_password):
                st.success("Successfully signed up as {}".format(new_username))
                st.write("Please go to the login page to log in.")  # Prompt user to go to login page
            else:
                st.error("Username already exists")
        else:
            st.error("Passwords do not match")

# Function to create a new user in the database
def create_user(username, password):
    hashed_password = hash_password(password)
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print("SQLite error:", e)
        return False

# Function to retrieve hashed password from the database
def get_stored_password(username):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        conn.close()
        if result:
            return result[0]
        else:
            return None
    except sqlite3.Error as e:
        print("SQLite error:", e)
        return None

# Function to hash the password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Render main page after successful login
#def render_main_page():
    # st.subheader("Main Page")

    # Sidebar navigation
   # st.sidebar.subheader("Navigation")
    
    # Button for Home page
    #home_button_clicked = st.sidebar.button("Machine learning decision system",key="home_button")

    # Button for About page
    #about_button_clicked = st.sidebar.button("Real time prediction",key="about_button")

    # Button for Contact page
    #contact_button_clicked = st.sidebar.button("Historical EDA metrics")


    #scheduler_button_clicked = st.sidebar.button("Maintenance Scheduler")

    # Sign Out button
    #if st.sidebar.button("Sign Out"):
    #    st.session_state.logged_in = False
    #    st.success("Successfully logged out")
    #    return
        
    #if about_button_clicked:
    #    csv_pred.csv_prediction()
    #elif contact_button_clicked:
    #    model_eval.model_evaluation()
    #else:
    #    model.model_run()

# Run the main function
if __name__ == "__main__":
    main()
