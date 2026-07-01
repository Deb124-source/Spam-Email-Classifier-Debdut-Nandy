import streamlit as st
import joblib


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Spamify AI",
    page_icon="📩",
    layout="centered"
)


# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# -----------------------------
# UI
# -----------------------------

st.title(" Spamify AI")
st.subheader("AI Powered Spam Email Classifier")

st.write(
    "Enter an email or message below and AI will detect whether it is spam or not."
)


# Input box

message = st.text_area(
    "Enter your message:",
    height=180,
    placeholder="Example: Congratulations! You won a free prize..."
)


# Button

if st.button(" Detect Spam"):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:

        # Transform text
        transformed = vectorizer.transform([message])


        # Prediction
        prediction = model.predict(transformed)[0]


        # Result UI

        if prediction == "spam":

            st.error(
                "🚨 This message is Spam"
            )

        else:

            st.success(
                "✅ This message is Not Spam"
            )
