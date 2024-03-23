import streamlit as st
from app import hash_password,get_stored_password

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
login()