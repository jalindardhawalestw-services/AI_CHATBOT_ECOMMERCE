import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Your AI Assistant", page_icon="🖖")

st.markdown(
    """
    <style>
    body {
        background-color: #fad0c4; /* Set your desired background color */
    }
    </style>
    """,
    unsafe_allow_html=True
)
logo = Image.open(r"STW-LOGO.png")
# st.image(logo)
col1, col2, col3 = st.columns(3)
with col1:
    st.write("")
with col2:
    st.image(logo, caption='STW Services')

with col3:
    st.write("")
    

intents = json.load(open('intent.json'))
tags = []
patterns = []

#  looping through all the intents and Identifying the patterns and greetings from the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# to extract the features from the patterns
vector = TfidfVectorizer()
patterns_scaled = vector.fit_transform(patterns)

# Model
Bot = LogisticRegression(max_iter=100000)
Bot.fit(patterns_scaled, tags)


# Identifying the tag for the input
def ChatBot(input_message):
    input_message = vector.transform([input_message])
    pred_tag = Bot.predict(input_message)[0]
    for intent in intents['intents']:
        if intent['tag'] == pred_tag:
            response = random.choice(intent['responses'])
            return response


st.markdown("<h1 style='text-align: center; color: red;'>Ecommerce AI ChatBot</h1>", unsafe_allow_html=True)

# Include FontAwesome CDN
st.markdown("""
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)

# AI Chatbot response with horizontally aligned product icons
with st.chat_message("assistant"):
    st.markdown("""
        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; display: flex; align-items: center;">
            <i class="fas fa-tshirt" style="font-size: 24px; color: #333333; margin-right: 10px;"></i>
            <span>Fashion</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; display: flex; align-items: center;">
            <i class="fas fa-shoe-prints" style="font-size: 24px; color: #333333; margin-right: 10px;"></i>
            <span>Footwear</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; display: flex; align-items: center;">
            <i class="fas fa-laptop" style="font-size: 24px; color: #333333; margin-right: 10px;"></i>
            <span>Laptops</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; display: flex; align-items: center;">
            <i class="fas fa-headphones-alt" style="font-size: 24px; color: #333333; margin-right: 10px;"></i>
            <span>Headphones</span>
        </div>
        """, unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = ChatBot(prompt)
    if response:
        response = f"AI ChatBot: {response}"  # Ensure the response starts with "AI ChatBot:"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
